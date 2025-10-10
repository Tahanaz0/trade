import threading
import pandas as pd
import numpy as np
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
        self.open_positions = []
    
    # ✅ IMPROVEMENT 1: Enhanced BOS Detection with Strength Measurement
    def is_bos(self, df, idx):
        """Detect Break of Structure with strength validation"""
        if idx < 5:
            return None, 0
        
        cur = df.iloc[idx]
        prev = df.iloc[idx-1]
        prev2 = df.iloc[idx-2]
        
        # Bull BOS with strength check
        if cur["high"] > prev["high"] and prev["high"] > prev2["high"]:
            # Calculate BOS strength (percentage move)
            strength = ((cur["high"] - prev2["high"]) / prev2["high"]) * 100
            
            # Require minimum strength and volume confirmation
            vol_ratio = cur["volume"] / df["volume"].iloc[idx-10:idx].mean()
            if strength > 0.5 and vol_ratio > 1.2:  # Strong BOS with volume
                return "bull", strength
        
        # Bear BOS with strength check
        if cur["low"] < prev["low"] and prev["low"] < prev2["low"]:
            strength = ((prev2["low"] - cur["low"]) / prev2["low"]) * 100
            vol_ratio = cur["volume"] / df["volume"].iloc[idx-10:idx].mean()
            if strength > 0.5 and vol_ratio > 1.2:
                return "bear", strength
        
        return None, 0
    
    # ✅ IMPROVEMENT 2: Enhanced Order Block Detection
    def detect_order_block(self, df, idx, bos_type):
        """Detect valid order block with better validation"""
        if idx < 10:
            return None
        
        # Look back for the last bearish candle before bull BOS (or vice versa)
        lookback = min(10, idx)
        
        for i in range(1, lookback):
            ob_candle = df.iloc[idx - i]
            
            if bos_type == "bull":
                # Find last strong bearish candle
                if ob_candle["close"] < ob_candle["open"]:
                    body_size = abs(ob_candle["open"] - ob_candle["close"])
                    candle_range = ob_candle["high"] - ob_candle["low"]
                    
                    # Strong bearish candle (body > 60% of range)
                    if body_size / candle_range > 0.6:
                        return (ob_candle["high"], ob_candle["low"], ob_candle["close"])
            
            elif bos_type == "bear":
                # Find last strong bullish candle
                if ob_candle["close"] > ob_candle["open"]:
                    body_size = abs(ob_candle["close"] - ob_candle["open"])
                    candle_range = ob_candle["high"] - ob_candle["low"]
                    
                    if body_size / candle_range > 0.6:
                        return (ob_candle["high"], ob_candle["low"], ob_candle["close"])
        
        return None
    
    # ✅ IMPROVEMENT 3: Market Regime Filter
    def get_market_regime(self, df, idx):
        """Determine if market is trending or ranging"""
        if idx < 50:
            return "unknown"
        
        recent_data = df.iloc[idx-50:idx]
        
        # Calculate ADX-like trend strength
        ema_fast = recent_data["close"].ewm(span=20).mean()
        ema_slow = recent_data["close"].ewm(span=50).mean()
        
        price_range = recent_data["high"].max() - recent_data["low"].min()
        avg_price = recent_data["close"].mean()
        volatility = (price_range / avg_price) * 100
        
        # Strong trend if EMAs clearly separated
        ema_separation = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
        
        if ema_separation > 2 and volatility > 5:
            return "trending"
        elif volatility < 3:
            return "ranging"
        else:
            return "transitioning"
    
    # ✅ IMPROVEMENT 4: Dynamic TP/SL based on volatility
    def calculate_dynamic_tp_sl(self, df, idx, atr, risk_per_unit, config):
        """Calculate adaptive TP/SL ratios based on market conditions"""
        market_regime = self.get_market_regime(df, idx)
        
        # Base multipliers
        tp_multiplier = config["tp_rr"]
        sl_multiplier = config["atr_sl_multiplier"]
        
        # Adjust based on market regime
        if market_regime == "trending":
            tp_multiplier *= 1.3  # Ride trends longer
            sl_multiplier *= 0.9  # Tighter stops
        elif market_regime == "ranging":
            tp_multiplier *= 0.8  # Quick profits in ranging
            sl_multiplier *= 1.1  # Wider stops for noise
        
        # Check recent volatility
        recent_atr = df["ATR14"].iloc[idx-10:idx].mean()
        current_atr = atr
        
        if current_atr > recent_atr * 1.5:  # High volatility spike
            sl_multiplier *= 1.2  # Wider stops
        
        return tp_multiplier, sl_multiplier
    
    # ✅ IMPROVEMENT 5: Time-based filter
    def is_valid_trading_time(self, timestamp):
        """Filter out low-liquidity hours"""
        hour = timestamp.hour
        
        # Avoid low liquidity times (example: midnight to 4 AM UTC)
        # Adjust based on your market observations
        if 0 <= hour < 4:
            return False
        
        return True
    
    # ✅ IMPROVEMENT 6: Multi-timeframe confirmation
    def check_higher_timeframe_alignment(self, symbol, timeframe, direction):
        """Check if higher timeframe supports the trade direction"""
        # This is a placeholder - you'd need to fetch higher TF data
        # For now, returning True. Implement if you want HTF confirmation
        return True
    
    def can_open_new_trade(self, config, signal_timestamp, symbol):
        """Check if we can open a new trade"""
        if not config["fixed_investment_mode"]:
            return True
        
        max_concurrent_trades = int(config["initial_capital"] / config["fixed_investment_amount"])
        
        open_positions_at_time = [
            pos for pos in self.open_positions
            if pos['entry_timestamp'] <= signal_timestamp < pos['close_timestamp']
        ]
        
        if len(open_positions_at_time) >= max_concurrent_trades:
            return False
        
        for pos in open_positions_at_time:
            if pos['symbol'] == symbol:
                return False
        
        return True
    
    def process_long_entry(self, df, i, symbol, capital, daily_trades, config):
        """Process long entry with enhanced filters"""
        candle = df.iloc[i]
        date = candle.name.date()
        signal_timestamp = candle.name.timestamp()
        
        # Daily trade limit check
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
            return None, capital
        
        # Concurrent position check
        if not self.can_open_new_trade(config, signal_timestamp, symbol):
            return None, capital
        
        # ✅ Time filter
        if not self.is_valid_trading_time(candle.name):
            return None, capital
        
        atr = candle["ATR14"]
        vol_avg = df["VOL_AVG14"].iloc[i]
        
        if pd.isna(atr) or pd.isna(vol_avg):
            return None, capital
        
        # ✅ Enhanced BOS check with strength
        bos, bos_strength = self.is_bos(df, i)
        if bos != "bull" or bos_strength < 0.5:
            return None, capital
        
        # EMA trend filter
        if candle["EMA50"] <= candle["EMA200"]:
            return None, capital
        
        # ✅ Market regime filter
        regime = self.get_market_regime(df, i)
        if regime == "ranging":
            return None, capital  # Skip ranging markets for trend strategy
        
        # ✅ Enhanced volume check
        vol_ratio = candle["volume"] / vol_avg
        if vol_ratio < config["volume_multiplier"]:
            return None, capital
        
        # ✅ Enhanced order block detection
        ob = self.detect_order_block(df, i, "bull")
        if ob is None:
            return None, capital
        
        ob_high, ob_low, ob_close = ob
        entry_idx = None
        
        # Find entry point with better validation
        for j in range(i+1, min(i+1+config["max_lookforward_bars"], len(df))):
            fc = df.iloc[j]
            
            # Price touches order block zone
            if fc["low"] <= ob_high + (atr * 0.25) and fc["high"] >= ob_low - (atr * 0.25):
                # Wait for bullish confirmation
                if fc["close"] > fc["open"] and fc["close"] > ob_close:
                    entry_idx = j
                    break
                if j+1 < len(df):
                    nc = df.iloc[j+1]
                    if nc["close"] > nc["open"] and nc["close"] > ob_close:
                        entry_idx = j+1
                        break
        
        if entry_idx is None:
            return None, capital
        
        return self.execute_trade(df, entry_idx, symbol, "LONG", ob_low, atr, capital, config, daily_trades, date)
    
    def process_short_entry(self, df, i, symbol, capital, daily_trades, config):
        """Process short entry with enhanced filters"""
        candle = df.iloc[i]
        date = candle.name.date()
        signal_timestamp = candle.name.timestamp()
        
        if daily_trades.get(date, 0) >= config["daily_trade_limit_per_coin"]:
            return None, capital
        
        if not self.can_open_new_trade(config, signal_timestamp, symbol):
            return None, capital
        
        # ✅ Time filter
        if not self.is_valid_trading_time(candle.name):
            return None, capital
        
        atr = candle["ATR14"]
        vol_avg = df["VOL_AVG14"].iloc[i]
        
        if pd.isna(atr) or pd.isna(vol_avg):
            return None, capital
        
        # ✅ Enhanced BOS check
        bos, bos_strength = self.is_bos(df, i)
        if bos != "bear" or bos_strength < 0.5:
            return None, capital
        
        if candle["EMA50"] >= candle["EMA200"]:
            return None, capital
        
        # ✅ Market regime filter
        regime = self.get_market_regime(df, i)
        if regime == "ranging":
            return None, capital
        
        # ✅ Enhanced volume check
        vol_ratio = candle["volume"] / vol_avg
        if vol_ratio < config["volume_multiplier"]:
            return None, capital
        
        # ✅ Enhanced order block detection
        ob = self.detect_order_block(df, i, "bear")
        if ob is None:
            return None, capital
        
        ob_high, ob_low, ob_close = ob
        entry_idx = None
        
        for j in range(i+1, min(i+1+config["max_lookforward_bars"], len(df))):
            fc = df.iloc[j]
            
            if fc["high"] >= ob_low - (atr * 0.25) and fc["low"] <= ob_high + (atr * 0.25):
                if fc["close"] < fc["open"] and fc["close"] < ob_close:
                    entry_idx = j
                    break
                if j+1 < len(df):
                    nc = df.iloc[j+1]
                    if nc["close"] < nc["open"] and nc["close"] < ob_close:
                        entry_idx = j+1
                        break
        
        if entry_idx is None:
            return None, capital
        
        return self.execute_trade(df, entry_idx, symbol, "SHORT", ob_high, atr, capital, config, daily_trades, date)
    
    def execute_trade(self, df, entry_idx, symbol, trade_type, ob_level, atr, capital, config, daily_trades, date):
        """Execute trade with dynamic TP/SL"""
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row["close"]
        entry_timestamp = entry_row.name.timestamp()
        
        # ✅ Get dynamic TP/SL multipliers
        tp_mult, sl_mult = self.calculate_dynamic_tp_sl(df, entry_idx, atr, 0, config)
        
        # Calculate SL
        if trade_type == "LONG":
            sl_price = min(ob_level - atr * 0.1, entry_price - atr * sl_mult)
            risk_per_unit = abs(entry_price - sl_price)
        else:
            sl_price = max(ob_level + atr * 0.1, entry_price + atr * sl_mult)
            risk_per_unit = abs(sl_price - entry_price)
        
        # Position sizing
        if config["fixed_investment_mode"]:
            invested = config["fixed_investment_amount"]
            position_size_units = invested / entry_price
        else:
            risk_amount = capital * config["risk_per_trade"]
            if risk_per_unit <= 0:
                return None, capital
            position_size_units = risk_amount / risk_per_unit
            invested = position_size_units * entry_price
            if invested > capital:
                scale_factor = capital / invested
                position_size_units *= scale_factor
                invested = capital
        
        # ✅ Calculate dynamic TP
        if trade_type == "LONG":
            tp_price = entry_price + (risk_per_unit * tp_mult)
        else:
            tp_price = entry_price - (risk_per_unit * tp_mult)
        
        result, pnl, close_idx = self.check_trade_outcome(df, entry_idx, trade_type, sl_price, tp_price, 
                                             position_size_units, entry_price, config)
        
        # Track position
        if config["fixed_investment_mode"]:
            if close_idx is not None:
                close_timestamp = df.iloc[close_idx].name.timestamp()
            else:
                close_timestamp = df.iloc[-1].name.timestamp()
            
            self.open_positions.append({
                'symbol': symbol,
                'entry_timestamp': entry_timestamp,
                'close_timestamp': close_timestamp,
                'invested': invested
            })
        
        # Capital update
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
        
        daily_trades[date] = daily_trades.get(date, 0) + 1
        
        return trade, capital
    
    def check_trade_outcome(self, df, entry_idx, trade_type, sl_price, tp_price, position_size, entry_price, config):
        """Check trade outcome"""
        result = "OPEN"
        pnl = 0.0
        close_idx = None
        
        for k in range(entry_idx+1, len(df)):
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
        
        if result == "OPEN" and len(df) > entry_idx:
            last_price = df.iloc[-1]["close"]
            if trade_type == "LONG":
                pnl = position_size * (last_price - entry_price)
            else:
                pnl = position_size * (entry_price - last_price)
        
        return result, pnl, close_idx
    
    def run_strategy(self):
        """Main strategy execution"""
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
        
        print(f"🚀 Starting Enhanced Backtest")
        print(f"📊 Initial Capital: ${capital}")
        print(f"🎯 Dynamic TP/SL: Enabled")
        print(f"🔍 Market Regime Filter: Active")
        print(f"⏰ Time Filter: Active")
        
        # Collect signals
        all_potential_signals = []
        
        for symbol in config["coins"]:
            print(f"   Processing {symbol}...")
            try:
                df = self.data_fetcher.fetch_ohlcv(symbol, config["timeframe"], days=days)
                if df.empty:
                    continue
                    
                df = self.data_fetcher.add_indicators(df)
                
                for i in range(10, len(df)-1):  # Start from 10 for better indicators
                    candle = df.iloc[i]
                    signal_timestamp = candle.name.timestamp()
                    
                    bos, strength = self.is_bos(df, i)
                    
                    if bos == "bull" and candle["EMA50"] > candle["EMA200"]:
                        all_potential_signals.append({
                            'symbol': symbol,
                            'timestamp': signal_timestamp,
                            'df': df,
                            'index': i,
                            'type': 'LONG',
                            'strength': strength
                        })
                    
                    if bos == "bear" and candle["EMA50"] < candle["EMA200"]:
                        all_potential_signals.append({
                            'symbol': symbol,
                            'timestamp': signal_timestamp,
                            'df': df,
                            'index': i,
                            'type': 'SHORT',
                            'strength': strength
                        })
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        all_potential_signals.sort(key=lambda x: x['timestamp'])
        print(f"\n📊 Found {len(all_potential_signals)} signals")
        
        # Process signals
        monthly_data = {}
        daily_trades_tracker = {}
        
        for signal in all_potential_signals:
            symbol = signal['symbol']
            df = signal['df']
            i = signal['index']
            signal_type = signal['type']
            
            if symbol not in daily_trades_tracker:
                daily_trades_tracker[symbol] = {}
            
            trade = None
            if signal_type == 'LONG':
                trade, capital = self.process_long_entry(df, i, symbol, capital, daily_trades_tracker[symbol], config)
            else:
                trade, capital = self.process_short_entry(df, i, symbol, capital, daily_trades_tracker[symbol], config)
            
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
        
        # Withdrawal logic
        if config["enable_withdrawal"] and not config["fixed_investment_mode"]:
            for month, data in monthly_data.items():
                if data["profit"] > 0:
                    withdraw = data["profit"] * config["withdraw_percentage"]
                    reinvest = data["profit"] * config["reinvest_percentage"]
                    total_withdrawn += withdraw
                    if config["compounding"]:
                        capital += reinvest
        
        if not config["compounding"] and not config["fixed_investment_mode"]:
            total_pnl = sum(t["pnl"] for t in self.trades_data)
            capital = config["initial_capital"] + total_pnl
        
        summary = self.calculate_summary(config, capital, total_withdrawn, total_invested_volume)
        self.performance_tracker.set_summary(summary)
        
        print(f"\n✅ Backtest Complete!")
        print(f"💰 Final Capital: ${capital:.2f}")
        print(f"📈 Net Profit: ${summary['net_profit']:.2f}")
        print(f"🎯 Win Rate: {summary['winrate']}%")
        
        self.is_running = False
    
    def calculate_summary(self, config, capital, total_withdrawn, total_invested_volume):
        """Calculate summary"""
        wins = sum(1 for t in self.trades_data if t["result"] == "WIN")
        losses = sum(1 for t in self.trades_data if t["result"] == "LOSS")
        total = wins + losses
        winrate = (wins / total * 100) if total > 0 else 0
        net_profit = (capital + total_withdrawn) - config["initial_capital"]
        
        return {
            "initial_capital": config["initial_capital"],
            "ending_capital": round(capital, 2),
            "total_withdrawn": round(total_withdrawn, 2),
            "total_invested_volume": round(total_invested_volume, 2),
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "winrate": round(winrate, 2),
            "net_profit": round(net_profit, 2)
        }
    
    def start_backtest(self):
        """Start backtest"""
        if not self.is_running:
            threading.Thread(target=self.run_strategy, daemon=True).start()
    
    def get_trades_data(self):
        return self.trades_data
    
    def get_open_signals(self):
        return self.open_signals
    
    def get_running_status(self):
        return self.is_running