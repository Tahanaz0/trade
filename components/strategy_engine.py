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
        self.open_positions = []  # Track open positions for concurrent trade limiting
    
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
    
    def can_open_new_trade(self, config, signal_timestamp, symbol):
        """
        ✅ Check if we can open a new trade based on:
        1. Total concurrent position limit
        2. Per-coin position limit (only 1 trade per coin at a time)
        """
        if not config["fixed_investment_mode"]:
            return True
        
        # Calculate max concurrent trades based on capital and fixed investment
        max_concurrent_trades = int(config["initial_capital"] / config["fixed_investment_amount"])
        
        # Count positions that are still open at signal_timestamp
        open_positions_at_time = [
            pos for pos in self.open_positions
            if pos['entry_timestamp'] <= signal_timestamp < pos['close_timestamp']
        ]
        
        # ✅ Check 1: Total concurrent limit
        if len(open_positions_at_time) >= max_concurrent_trades:
            print(f"   ⚠️ Max concurrent positions reached ({len(open_positions_at_time)}/{max_concurrent_trades}). Skipping new trade.")
            return False
        
        # ✅ Check 2: Per-coin limit (only 1 trade per coin)
        for pos in open_positions_at_time:
            if pos['symbol'] == symbol:
                print(f"   ⚠️ {symbol} already has an open position. Skipping new trade.")
                return False
        
        return True
    
    def process_long_entry(self, df, i, symbol, capital, daily_trades, config):
        """Process long entry signal"""
        candle = df.iloc[i]
        date = candle.name.date()
        signal_timestamp = candle.name.timestamp()
        
        # ✅ Check daily trade limit per coin
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
            return None, capital
        
        # ✅ Check concurrent position limit (total + per-coin)
        if not self.can_open_new_trade(config, signal_timestamp, symbol):
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
        signal_timestamp = candle.name.timestamp()
        
        # ✅ Check daily trade limit per coin
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
            return None, capital
        
        # ✅ Check concurrent position limit (total + per-coin)
        if not self.can_open_new_trade(config, signal_timestamp, symbol):
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
        entry_timestamp = entry_row.name.timestamp()
        
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
        
        # ✅ Track open position for concurrent limit
        if config["fixed_investment_mode"]:
            if close_idx is not None:
                close_timestamp = df.iloc[close_idx].name.timestamp()
            else:
                # If trade is still open, set close_timestamp to far future
                close_timestamp = float('inf')
            
            self.open_positions.append({
                'symbol': symbol,
                'entry_timestamp': entry_timestamp,
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
            "timestamp": entry_timestamp
        }
        
        # Count current open positions
        current_open = sum(
            1 for pos in self.open_positions
            if pos['entry_timestamp'] <= entry_timestamp < pos['close_timestamp']
        )
        
        print(f"✅ Trade: {symbol} {trade_type} | Open Positions: {current_open} | Invested: ${invested:.2f} | PnL: ${pnl:.4f}")
        
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
        """Main strategy execution method - processes all signals chronologically"""
        self.is_running = True
        self.trades_data = []
        self.open_signals = []
        self.open_positions = []
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
            print(f"🔒 Per-Coin Limit: 1 trade at a time")
        
        print(f"💵 Withdrawal Enabled: {config['enable_withdrawal']}")
        if config['enable_withdrawal']:
            print(f"📤 Withdraw %: {config['withdraw_percentage'] * 100}%")
            print(f"🔄 Reinvest %: {config['reinvest_percentage'] * 100}%")
        
        # ✅ Step 1: Collect all potential signals from all coins
        print("\n🔍 Collecting signals from all coins...")
        all_potential_signals = []
        
        for symbol in config["coins"]:
            print(f"   Processing {symbol}...")
            try:
                df = self.data_fetcher.fetch_ohlcv(symbol, config["timeframe"], days=days)
                if df.empty:
                    print(f"   ❌ No data for {symbol}")
                    continue
                    
                df = self.data_fetcher.add_indicators(df)
                
                for i in range(2, len(df)-1):
                    candle = df.iloc[i]
                    signal_timestamp = candle.name.timestamp()
                    
                    # Check for potential long signal
                    if self.is_bos(df, i) == "bull" and candle["EMA50"] > candle["EMA200"]:
                        all_potential_signals.append({
                            'symbol': symbol,
                            'timestamp': signal_timestamp,
                            'df': df,
                            'index': i,
                            'type': 'LONG'
                        })
                    
                    # Check for potential short signal
                    if self.is_bos(df, i) == "bear" and candle["EMA50"] < candle["EMA200"]:
                        all_potential_signals.append({
                            'symbol': symbol,
                            'timestamp': signal_timestamp,
                            'df': df,
                            'index': i,
                            'type': 'SHORT'
                        })
                        
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")
                continue
        
        # ✅ Step 2: Sort all signals chronologically
        all_potential_signals.sort(key=lambda x: x['timestamp'])
        print(f"\n📊 Found {len(all_potential_signals)} potential signals across all coins")
        
        # ✅ Step 3: Process signals in chronological order
        print("\n🔄 Processing signals chronologically...\n")
        monthly_data = {}
        daily_trades_tracker = {}
        
        for signal in all_potential_signals:
            symbol = signal['symbol']
            df = signal['df']
            i = signal['index']
            signal_type = signal['type']
            
            # Get daily trades for this symbol
            if symbol not in daily_trades_tracker:
                daily_trades_tracker[symbol] = {}
            
            trade = None
            if signal_type == 'LONG':
                trade, capital = self.process_long_entry(
                    df, i, symbol, capital, daily_trades_tracker[symbol], config
                )
            else:
                trade, capital = self.process_short_entry(
                    df, i, symbol, capital, daily_trades_tracker[symbol], config
                )
            
            if trade:
                self.trades_data.append(trade)
                total_invested_volume += trade["invested"]
                self.performance_tracker.update_performance(trade)
                
                month_key = trade['raw_date'].strftime('%Y-%m')
                if month_key not in monthly_data:
                    monthly_data[month_key] = {"profit": 0.0, "trades": 0}
                monthly_data[month_key]["profit"] += trade['pnl']
                monthly_data[month_key]["trades"] += 1
                
                if trade["result"] == "OPEN":
                    self.open_signals.append(trade)
        
        print(f"\n📊 Processed {len(self.trades_data)} trades chronologically")
        
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