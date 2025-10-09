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
    
    def process_long_entry(self, df, i, symbol, capital, daily_trades, config):
        """Process long entry signal"""
        candle = df.iloc[i]
        date = candle.name.date()
        
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
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
        
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
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
        
        # ✅ FIXED: Position sizing logic with proper compounding
        if config["fixed_investment_mode"]:
            invested = config["fixed_investment_amount"]
            position_size_units = invested / entry_price
        else:
            # Risk amount based on current capital
            risk_amount = capital * config["risk_per_trade"]

            if risk_per_unit <= 0:
                print(f"❌ Invalid risk per unit for {symbol}: {risk_per_unit}")
                return None, capital

            # Calculate position size based on RISK
            position_size_units = risk_amount / risk_per_unit
            invested = position_size_units * entry_price

            # Capital management
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
        result, pnl = self.check_trade_outcome(df, entry_idx, trade_type, sl_price, tp_price, 
                                             position_size_units, entry_price, config)
        
        # ✅ FIXED: Capital Update Logic for Compounding
        if config["fixed_investment_mode"]:
            # Fixed mode: Always update capital with PnL
            capital += pnl
        else:
            if config["compounding"]:
                # Compounding mode: Update capital with PnL for next trade
                capital += pnl
            else:
                # Non-compounding mode: Capital remains same for position sizing
                # PnL is tracked separately but doesn't affect position sizing
                pass
        
        # Create trade record
        trade_date = entry_row.name
        trade = {
            "symbol": symbol,
            "date": trade_date.strftime("%Y-%m-%d %H:%M"),
            "type": trade_type,
            "entry": round(entry_price, 8),
            "sl": round(sl_price, 8),
            "tp": round(tp_price, 8),
            "invested": round(invested, 8),
            "position_size": round(position_size_units, 8),
            "result": result,
            "pnl": round(pnl, 8),
            "raw_date": trade_date,
            "timestamp": trade_date.timestamp()
        }
        
        print(f"✅ Trade: {symbol} {trade_type} | Capital: ${capital:.2f} | Risk: ${capital * config['risk_per_trade']:.2f} | Invested: ${invested:.2f} | PnL: ${pnl:.4f}")
        
        daily_trades[date] = daily_trades.get(date, 0) + 1
        
        return trade, capital

    def check_trade_outcome(self, df, entry_idx, trade_type, sl_price, tp_price, position_size, entry_price, config):
        """Check if trade hits SL or TP"""
        result = "OPEN"
        pnl = 0.0
        
        for k in range(entry_idx+1, min(len(df), entry_idx + config["max_lookforward_bars"])):
            sc = df.iloc[k]
            
            if trade_type == "LONG":
                if sc["low"] <= sl_price:
                    result = "LOSS"
                    pnl = position_size * (sl_price - entry_price)
                    break
                elif sc["high"] >= tp_price:
                    result = "WIN"
                    pnl = position_size * (tp_price - entry_price)
                    break
            else:
                if sc["high"] >= sl_price:
                    result = "LOSS"
                    pnl = position_size * (entry_price - sl_price)
                    break
                elif sc["low"] <= tp_price:
                    result = "WIN"
                    pnl = position_size * (entry_price - tp_price)
                    break
        
        return result, pnl
    
    def run_strategy(self):
        """Main strategy execution method"""
        self.is_running = True
        self.trades_data = []
        self.open_signals = []
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
        print(f"💰 Expected Risk Amount: ${capital * config['risk_per_trade']:.2f} per trade")
        print(f"💵 Withdrawal Enabled: {config['enable_withdrawal']}")
        if config['enable_withdrawal']:
            print(f"📤 Withdraw %: {config['withdraw_percentage'] * 100}%")
            print(f"🔄 Reinvest %: {config['reinvest_percentage'] * 100}%")
        
        # Store monthly data for withdrawal calculation
        monthly_data = {}
        
        # Collect all trades first, then sort by date
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
                        
                        # Track monthly data for withdrawal calculation
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
                        
                        # Track monthly data for withdrawal calculation
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
        
        # ✅ SORT TRADES BY DATE (Chronological Order)
        if all_trades:
            all_trades.sort(key=lambda x: x['timestamp'])
            self.trades_data = all_trades
            print(f"📅 Sorted {len(self.trades_data)} trades chronologically")
            
            # Print first few trades to verify sorting and investment
            print("First 5 trades after sorting:")
            for i, trade in enumerate(self.trades_data[:5]):
                print(f"  {i+1}. {trade['symbol']} - {trade['date']} - Invested: ${trade['invested']:.2f} - PnL: ${trade['pnl']:.4f}")
        else:
            self.trades_data = []
            print("❌ No trades executed")
        
        # ✅ FIXED: Monthly Withdrawal Logic (Compounding compatible)
        if config["enable_withdrawal"] and not config["fixed_investment_mode"]:
            print(f"\n💰 Processing Monthly Withdrawals...")
            print(f"Monthly Data Summary: {monthly_data}")
            
            if monthly_data:
                for month, data in monthly_data.items():
                    monthly_profit = data["profit"]
                    if monthly_profit > 0:
                        withdraw_amount = monthly_profit * config["withdraw_percentage"]
                        reinvest_amount = monthly_profit * config["reinvest_percentage"]
                        
                        # Verify the math
                        expected_total = withdraw_amount + reinvest_amount
                        if abs(expected_total - monthly_profit) > 0.01:
                            reinvest_amount = monthly_profit - withdraw_amount
                        
                        total_withdrawn += withdraw_amount
                        
                        # ✅ FIXED: Only add reinvest amount if compounding is enabled
                        if config["compounding"]:
                            capital += reinvest_amount
                        
                        print(f"📅 Month {month}:")
                        print(f"   Monthly Profit: ${monthly_profit:.2f}")
                        print(f"   Withdrawn ({config['withdraw_percentage']*100}%): ${withdraw_amount:.2f}")
                        print(f"   Reinvested ({config['reinvest_percentage']*100}%): ${reinvest_amount:.2f}")
                        print(f"   Total Withdrawn so far: ${total_withdrawn:.2f}")
                        print(f"   Capital after processing: ${capital:.2f}")
                    else:
                        print(f"📅 Month {month}: No profit (${monthly_profit:.2f}) - No withdrawal")
            else:
                print("❌ No monthly data available for withdrawal calculation")
        else:
            withdrawal_reason = "Fixed investment mode active" if config["fixed_investment_mode"] else "Withdrawal disabled in settings"
            print(f"ℹ️ Withdrawal skipped: {withdrawal_reason}")
        
        # ✅ FIXED: Final capital calculation for non-compounding mode
        if not config["compounding"] and not config["fixed_investment_mode"]:
            total_pnl = sum(trade["pnl"] for trade in self.trades_data)
            capital = config["initial_capital"] + total_pnl
            print(f"📊 Non-compounding mode calculation:")
            print(f"   Initial Capital: ${config['initial_capital']}")
            print(f"   Total PnL from trades: ${total_pnl:.2f}")
            print(f"   Final Capital: ${capital:.2f}")
        
        # Calculate final summary
        summary = self.calculate_summary(config, capital, total_withdrawn, total_invested_volume)
        self.performance_tracker.set_summary(summary)
        
        print(f"\n🎯 Backtest Completed!")
        print(f"📊 Total Trades: {len(self.trades_data)}")
        print(f"💰 Final Capital: ${capital:.2f}")
        print(f"💵 Total Withdrawn: ${total_withdrawn:.2f}")
        print(f"📈 Total Invested Volume: ${total_invested_volume:.2f}")
        print(f"📈 Open Signals: {len(self.open_signals)}")
        print(f"💸 Net Profit: ${summary['net_profit']:.2f}")
        print(f"🏆 Win Rate: {summary['winrate']}%")
        
        # ✅ ADDED: PnL verification
        total_calculated_pnl = sum(trade["pnl"] for trade in self.trades_data)
        expected_net_profit = capital - config["initial_capital"]
        print(f"\n📊 PnL Verification:")
        print(f"   Total PnL from all trades: ${total_calculated_pnl:.2f}")
        print(f"   Expected Net Profit: ${expected_net_profit:.2f}")
        print(f"   Reported Net Profit: ${summary['net_profit']:.2f}")
        
        if abs(total_calculated_pnl - expected_net_profit) > 0.01:
            print(f"⚠️ WARNING: PnL mismatch! Difference: ${abs(total_calculated_pnl - expected_net_profit):.4f}")
        else:
            print(f"✅ PnL calculation is correct!")
        
        # Investment analysis
        if self.trades_data:
            avg_investment = total_invested_volume / len(self.trades_data)
            max_investment = max(trade["invested"] for trade in self.trades_data)
            min_investment = min(trade["invested"] for trade in self.trades_data)
            
            print(f"\n📊 Investment Analysis:")
            print(f"   Average Investment per Trade: ${avg_investment:.2f}")
            print(f"   Maximum Investment: ${max_investment:.2f}")
            print(f"   Minimum Investment: ${min_investment:.2f}")
            print(f"   Expected Risk per Trade: ${config['initial_capital'] * config['risk_per_trade']:.2f}")
            
            # Show how investment grew over time (compounding effect)
            if config["compounding"] and len(self.trades_data) > 10:
                first_5_avg = sum(t["invested"] for t in self.trades_data[:5]) / 5
                last_5_avg = sum(t["invested"] for t in self.trades_data[-5:]) / 5
                growth_pct = ((last_5_avg - first_5_avg) / first_5_avg) * 100
                print(f"   Compounding Growth: First 5 avg: ${first_5_avg:.2f} → Last 5 avg: ${last_5_avg:.2f} ({growth_pct:+.1f}%)")
        
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