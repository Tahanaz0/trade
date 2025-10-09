import threading
import pandas as pd
from datetime import datetime
from components.data_fetcher import DataFetcher

class StrategyEngine:
    def __init__(self, config_manager, performance_tracker):
        self.config_manager = config_manager
        self.performance_tracker = performance_tracker
        self.data_fetcher = DataFetcher()
        self.is_running = False
        self.trades_data = []
        self.open_signals = []
        self.open_positions = []  # ✅ NEW: Track open positions for concurrent trade limiting
    
    def is_bos(self, df, idx):
        """Detect Break of Structure"""
        if idx < 3:
            return None
        
        cur = df.iloc[idx]
        prev = df.iloc[idx-1]
        prev2 = df.iloc[idx-2]
        
        if cur["high"] > prev["high"] and prev["high"] > prev2["high"]:
            return "bull"
        if cur["low"] < prev["low"] and prev["low"] < prev2["low"]:
            return "bear"
        return None
    
    def detect_order_block(self, df, idx):
        """Detect order block"""
        if idx - 1 < 0:
            return None
        ob_candle = df.iloc[idx-1]
        return (ob_candle["high"], ob_candle["low"])
    
    def can_open_new_trade(self, config, current_index, df):
        """
        ✅ NEW: Check if we can open a new trade based on concurrent position limits
        """
        if not config["fixed_investment_mode"]:
            return True
        
        # Calculate max concurrent trades based on capital and fixed investment
        max_concurrent_trades = int(config["initial_capital"] / config["fixed_investment_amount"])
        
        # Close positions that have reached SL/TP by current index
        current_timestamp = df.iloc[current_index].name.timestamp()
        
        # Update open positions - remove closed ones
        self.open_positions = [
            pos for pos in self.open_positions 
            if pos['close_timestamp'] > current_timestamp
        ]
        
        current_open = len(self.open_positions)
        
        if current_open >= max_concurrent_trades:
            return False
        
        return True
    
    def process_long_entry(self, df, i, symbol, capital, daily_trades, config):
        """Process long entry signal"""
        candle = df.iloc[i]
        date = candle.name.date()
        
        # ✅ Check daily trade limit per coin
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
            return None, capital
        
        # ✅ NEW: Check concurrent position limit (for fixed investment mode)
        if not self.can_open_new_trade(config, i, df):
            print(f"   ⚠️ Max concurrent positions reached ({len(self.open_positions)}). Skipping new trade.")
            return None, capital
        
        atr = candle["ATR14"]
        vol_avg = df["VOL_AVG14"].iloc[i]
        
        if pd.isna(atr) or pd.isna(vol_avg) or candle["volume"] < config["volume_multiplier"] * vol_avg:
            return None, capital
        
        bos = self.is_bos(df, i)
        if bos != "bull" or candle["EMA50"] <= candle["EMA200"]:
            return None, capital
        
        ob = self.detect_order_block(df, i)
        if ob is None:
            return None, capital
        
        ob_high, ob_low = ob
        entry_idx = None
        
        # Find entry point
        for j in range(i+1, min(i+1+config["max_lookforward_bars"], len(df))):
            fc = df.iloc[j]
            if fc["low"] <= ob_high + (atr * 0.25) and fc["high"] >= ob_low - (atr * 0.25):
                if fc["close"] > fc["open"]:
                    entry_idx = j
                    break
                if j+1 < len(df):
                    nc = df.iloc[j+1]
                    if nc["close"] > nc["open"]:
                        entry_idx = j+1
                        break
        
        if entry_idx is None:
            return None, capital
        
        return self.execute_trade(df, entry_idx, symbol, "LONG", ob_low, atr, capital, config, daily_trades, date)
    
    def process_short_entry(self, df, i, symbol, capital, daily_trades, config):
        """Process short entry signal"""
        candle = df.iloc[i]
        date = candle.name.date()
        
        # ✅ Check daily trade limit per coin
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
            return None, capital
        
        # ✅ NEW: Check concurrent position limit (for fixed investment mode)
        if not self.can_open_new_trade(config, i, df):
            print(f"   ⚠️ Max concurrent positions reached ({len(self.open_positions)}). Skipping new trade.")
            return None, capital
        
        atr = candle["ATR14"]
        vol_avg = df["VOL_AVG14"].iloc[i]
        
        if pd.isna(atr) or pd.isna(vol_avg) or candle["volume"] < config["volume_multiplier"] * vol_avg:
            return None, capital
        
        bos = self.is_bos(df, i)
        if bos != "bear" or candle["EMA50"] >= candle["EMA200"]:
            return None, capital
        
        ob = self.detect_order_block(df, i)
        if ob is None:
            return None, capital
        
        ob_high, ob_low = ob
        entry_idx = None
        
        # Find entry point
        for j in range(i+1, min(i+1+config["max_lookforward_bars"], len(df))):
            fc = df.iloc[j]
            if fc["high"] >= ob_low - (atr * 0.25) and fc["low"] <= ob_high + (atr * 0.25):
                if fc["close"] < fc["open"]:
                    entry_idx = j
                    break
                if j+1 < len(df):
                    nc = df.iloc[j+1]
                    if nc["close"] < nc["open"]:
                        entry_idx = j+1
                        break
        
        if entry_idx is None:
            return None, capital
        
        return self.execute_trade(df, entry_idx, symbol, "SHORT", ob_high, atr, capital, config, daily_trades, date)
    
    def execute_trade(self, df, entry_idx, symbol, trade_type, ob_level, atr, capital, config, daily_trades, date):
        """Execute a trade and calculate results"""
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row["close"]
        
        # Calculate SL and position size
        if trade_type == "LONG":
            sl_price = min(ob_level - atr * 0.1, entry_price - atr * config["atr_sl_multiplier"])
            risk_per_unit = abs(entry_price - sl_price)
        else:
            sl_price = max(ob_level + atr * 0.1, entry_price + atr * config["atr_sl_multiplier"])
            risk_per_unit = abs(sl_price - entry_price)
        
        # Position sizing logic
        if config["fixed_investment_mode"]:
            invested = config["fixed_investment_amount"]
            position_size_units = invested / entry_price
        else:
            risk_amount = capital * config["risk_per_trade"]

            if risk_per_unit <= 0:
                print(f"❌ Invalid risk per unit for {symbol}: {risk_per_unit}")
                return None, capital

            position_size_units = risk_amount / risk_per_unit
            invested = position_size_units * entry_price

            if invested > capital:
                print(f"   ⚠️ Adjusting: Invested (${invested:.2f}) > Capital (${capital:.2f})")
                scale_factor = capital / invested
                position_size_units *= scale_factor
                invested = capital
        
        # Calculate TP
        if trade_type == "LONG":
            tp_price = entry_price + (risk_per_unit * config["tp_rr"])
        else:
            tp_price = entry_price - (risk_per_unit * config["tp_rr"])
        
        # Check trade outcome
        result, pnl, close_idx = self.check_trade_outcome(df, entry_idx, trade_type, sl_price, tp_price, 
                                             position_size_units, entry_price, config)
        
        # ✅ NEW: Track open position for concurrent limit
        if config["fixed_investment_mode"] and close_idx is not None:
            close_timestamp = df.iloc[close_idx].name.timestamp()
            self.open_positions.append({
                'symbol': symbol,
                'entry_timestamp': entry_row.name.timestamp(),
                'close_timestamp': close_timestamp,
                'invested': invested
            })
        
        # Capital Update Logic
        if config["fixed_investment_mode"]:
            capital += pnl
        else:
            if config["compounding"]:
                capital += pnl
        
        # Create trade record
        trade_date = entry_row.name
        close_date = df.iloc[close_idx].name if close_idx is not None else None
        
        trade = {
            "symbol": symbol,
            "date": trade_date.strftime("%Y-%m-%d %H:%M"),
            "close_date": close_date.strftime("%Y-%m-%d %H:%M") if close_date else "OPEN",
            "type": trade_type,
            "entry": round(entry_price, 8),
            "sl": round(sl_price, 8),
            "tp": round(tp_price, 8),
            "invested": round(invested, 8),
            "position_size": round(position_size_units, 8),
            "result": result,
            "pnl": round(pnl, 8),
            "raw_date": trade_date,
            "raw_close_date": close_date,
            "timestamp": trade_date.timestamp()
        }
        
        print(f"✅ Trade: {symbol} {trade_type} | Open Positions: {len(self.open_positions)} | Invested: ${invested:.2f} | PnL: ${pnl:.4f}")
        
        daily_trades[date] = daily_trades.get(date, 0) + 1
        
        return trade, capital

    def check_trade_outcome(self, df, entry_idx, trade_type, sl_price, tp_price, position_size, entry_price, config):
        """Check if trade hits SL or TP"""
        result = "OPEN"
        pnl = 0.0
        close_idx = None
        
        for k in range(entry_idx+1, min(len(df), entry_idx + config["max_lookforward_bars"])):
            sc = df.iloc[k]
            
            if trade_type == "LONG":
                if sc["low"] <= sl_price:
                    result = "LOSS"
                    pnl = position_size * (sl_price - entry_price)
                    close_idx = k
                    break
                elif sc["high"] >= tp_price:
                    result = "WIN"
                    pnl = position_size * (tp_price - entry_price)
                    close_idx = k
                    break
            else:
                if sc["high"] >= sl_price:
                    result = "LOSS"
                    pnl = position_size * (entry_price - sl_price)
                    close_idx = k
                    break
                elif sc["low"] <= tp_price:
                    result = "WIN"
                    pnl = position_size * (entry_price - tp_price)
                    close_idx = k
                    break
        
        return result, pnl, close_idx
    
    def run_strategy(self):
        """Main strategy execution method"""
        self.is_running = True
        self.trades_data = []
        self.open_signals = []
        self.open_positions = []  # ✅ Reset open positions
        self.performance_tracker.reset()
        
        config = self.config_manager.get_config()
        capital = config["initial_capital"]
        total_withdrawn = 0.0
        total_invested_volume = 0.0
        days = config["testing_months"] * 30
        
        print(f"🚀 Starting backtest with {len(config['coins'])} coins for {days} days")
        print(f"📊 Initial Capital: ${capital}")
        print(f"📈 Compounding: {config['compounding']}")
        print(f"🎯 Risk per Trade: {config['risk_per_trade'] * 100}%")
        
        if config["fixed_investment_mode"]:
            max_concurrent = int(config["initial_capital"] / config["fixed_investment_amount"])
            print(f"💰 Fixed Investment Mode: ${config['fixed_investment_amount']} per trade")
            print(f"🔢 Max Concurrent Positions: {max_concurrent}")
        
        print(f"💵 Withdrawal Enabled: {config['enable_withdrawal']}")
        if config['enable_withdrawal']:
            print(f"📤 Withdraw %: {config['withdraw_percentage'] * 100}%")
            print(f"🔄 Reinvest %: {config['reinvest_percentage'] * 100}%")
        
        monthly_data = {}
        all_trades = []
        
        for symbol in config["coins"]:
            print(f"\n🔍 Processing {symbol} ...")
            try:
                df = self.data_fetcher.fetch_ohlcv(symbol, config["timeframe"], days=days)
                if df.empty:
                    print(f"❌ No data for {symbol}")
                    continue
                    
                df = self.data_fetcher.add_indicators(df)
                print(f"📈 Loaded {len(df)} candles for {symbol}")
                
                daily_trades = {}
                trades_for_symbol = 0
                
                for i in range(2, len(df)-1):
                    trade = None
                    
                    # Check for long entries
                    trade, capital = self.process_long_entry(df, i, symbol, capital, daily_trades, config)
                    if trade:
                        all_trades.append(trade)
                        total_invested_volume += trade["invested"]
                        self.performance_tracker.update_performance(trade)
                        
                        month_key = trade['raw_date'].strftime('%Y-%m')
                        if month_key not in monthly_data:
                            monthly_data[month_key] = {"profit": 0.0, "trades": 0}
                        monthly_data[month_key]["profit"] += trade['pnl']
                        monthly_data[month_key]["trades"] += 1
                        
                        if trade["result"] == "OPEN":
                            self.open_signals.append(trade)
                        trades_for_symbol += 1
                        continue
                    
                    # Check for short entries
                    trade, capital = self.process_short_entry(df, i, symbol, capital, daily_trades, config)
                    if trade:
                        all_trades.append(trade)
                        total_invested_volume += trade["invested"]
                        self.performance_tracker.update_performance(trade)
                        
                        month_key = trade['raw_date'].strftime('%Y-%m')
                        if month_key not in monthly_data:
                            monthly_data[month_key] = {"profit": 0.0, "trades": 0}
                        monthly_data[month_key]["profit"] += trade['pnl']
                        monthly_data[month_key]["trades"] += 1
                        
                        if trade["result"] == "OPEN":
                            self.open_signals.append(trade)
                        trades_for_symbol += 1
                
                print(f"✅ {symbol}: {trades_for_symbol} trades executed")
                        
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Sort trades by date
        if all_trades:
            all_trades.sort(key=lambda x: x['timestamp'])
            self.trades_data = all_trades
            print(f"📅 Sorted {len(self.trades_data)} trades chronologically")
        else:
            self.trades_data = []
            print("❌ No trades executed")
        
        # Monthly Withdrawal Logic
        if config["enable_withdrawal"] and not config["fixed_investment_mode"]:
            print(f"\n💰 Processing Monthly Withdrawals...")
            
            if monthly_data:
                for month, data in monthly_data.items():
                    monthly_profit = data["profit"]
                    if monthly_profit > 0:
                        withdraw_amount = monthly_profit * config["withdraw_percentage"]
                        reinvest_amount = monthly_profit * config["reinvest_percentage"]
                        
                        expected_total = withdraw_amount + reinvest_amount
                        if abs(expected_total - monthly_profit) > 0.01:
                            reinvest_amount = monthly_profit - withdraw_amount
                        
                        total_withdrawn += withdraw_amount
                        
                        if config["compounding"]:
                            capital += reinvest_amount
                        
                        print(f"📅 Month {month}: Withdrawn ${withdraw_amount:.2f}, Reinvested ${reinvest_amount:.2f}")
        
        # Final capital calculation for non-compounding mode
        if not config["compounding"] and not config["fixed_investment_mode"]:
            total_pnl = sum(trade["pnl"] for trade in self.trades_data)
            capital = config["initial_capital"] + total_pnl
        
        # Calculate final summary
        summary = self.calculate_summary(config, capital, total_withdrawn, total_invested_volume)
        self.performance_tracker.set_summary(summary)
        
        print(f"\n🎯 Backtest Completed!")
        print(f"📊 Total Trades: {len(self.trades_data)}")
        print(f"💰 Final Capital: ${capital:.2f}")
        print(f"💵 Total Withdrawn: ${total_withdrawn:.2f}")
        print(f"💸 Net Profit: ${summary['net_profit']:.2f}")
        print(f"🏆 Win Rate: {summary['winrate']}%")
        
        self.is_running = False
    
    def calculate_summary(self, config, capital, total_withdrawn, total_invested_volume):
        """Calculate performance summary"""
        wins = sum(1 for t in self.trades_data if t["result"] == "WIN")
        losses = sum(1 for t in self.trades_data if t["result"] == "LOSS")
        total_trades = wins + losses
        winrate = (wins / total_trades * 100) if total_trades > 0 else 0
        net_profit = (capital + total_withdrawn) - config["initial_capital"]
        
        return {
            "initial_capital": config["initial_capital"],
            "ending_capital": round(capital, 2),
            "total_withdrawn": round(total_withdrawn, 2),
            "total_invested_volume": round(total_invested_volume, 2),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "winrate": round(winrate, 2),
            "net_profit": round(net_profit, 2)
        }
    
    def start_backtest(self):
        """Start backtest in separate thread"""
        if not self.is_running:
            print("🎬 Starting backtest in separate thread...")
            threading.Thread(target=self.run_strategy, daemon=True).start()
        else:
            print("⏳ Backtest is already running...")
    
    def get_trades_data(self):
        """Get trades data sorted by date"""
        return self.trades_data
    
    def get_open_signals(self):
        """Get open signals"""
        return self.open_signals
    
    def get_running_status(self):
        """Get running status"""
        return self.is_running