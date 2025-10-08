import ccxt
import pandas as pd
import pytz
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
import threading
from collections import defaultdict

# ------------------------------
# GLOBAL STATE
# ------------------------------
app = Flask(__name__)
trades_data = []
summary_data = {}
open_signals = []
is_running = False
coin_performance = {}
time_performance = {}

# Default settings
config = {
    "initial_capital": 50,
    "risk_per_trade": 0.01,
    # "coins": [
    #     "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    #     "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
    #     "DOGE/USDT", "LTC/USDT", "ATOM/USDT", "ETC/USDT", "XLM/USDT",
    #     "BCH/USDT", "FIL/USDT", "ALGO/USDT", "ICP/USDT", "EOS/USDT",
    #     "AAVE/USDT", "XTZ/USDT", "THETA/USDT", "TRX/USDT", "VET/USDT",
    #     "XMR/USDT", "SAND/USDT", "MANA/USDT", "NEAR/USDT", "FTM/USDT",
    #     "EGLD/USDT", "HBAR/USDT", "AXS/USDT", "FTT/USDT", "GRT/USDT",
    #     "MKR/USDT", "ZEC/USDT", "DASH/USDT", "ENJ/USDT", "COMP/USDT",
    #     "SNX/USDT", "BAT/USDT", "CHZ/USDT", "SUSHI/USDT", "CRV/USDT",
    #     "1INCH/USDT", "REN/USDT", "KAVA/USDT", "ANKR/USDT", "OCEAN/USDT"
    # ],
    # "coins": [
    # "HBAR/USDT", "ENJ/USDT", "GRT/USDT", "ZEC/USDT", "FTM/USDT",
    # "EGLD/USDT", "SNX/USDT", "NEAR/USDT", "DOGE/USDT", "SAND/USDT",
    # "MANA/USDT", "CHZ/USDT", "LINK/USDT", "CRV/USDT", "DOT/USDT",
    # "1INCH/USDT", "ALGO/USDT", "FIL/USDT", "OCEAN/USDT", "TRX/USDT",
    # "XLM/USDT", "VET/USDT", "ADA/USDT", "XMR/USDT", "SOL/USDT",
    # "BNB/USDT", "BTC/USDT"
    # ],
    "coins":["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "AVAX/USDT", "XRP/USDT", "LINK/USDT"],
    # coins = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "AVAX/USDT", "XRP/USDT", "LINK/USDT"]

    "timeframe": "1h",
    "daily_trade_limit_per_coin": 5,
    "withdraw_percentage": 0.30,
    "reinvest_percentage": 0.70,
    "volume_multiplier": 1.5,
    "atr_sl_multiplier": 1.0,
    "tp_rr": 3,
    "max_lookforward_bars": 240,
    "testing_months": 1,
    "compounding": True,
    "enable_withdrawal": True,
    "fixed_investment_mode": False,
    "fixed_investment_amount": 10
}

# ------------------------------
# BINANCE CONNECTION
# ------------------------------
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# ------------------------------
# HELPERS
# ------------------------------
def fetch_ohlcv(symbol, days=30):
    since = exchange.parse8601((datetime.now().astimezone(pytz.UTC) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    all_ohlcv = []
    limit = 1000
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=config["timeframe"], since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Karachi")
    df.set_index("timestamp", inplace=True)
    return df

def add_indicators(df):
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
    df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()
    df["VOL_AVG14"] = df["volume"].rolling(14).mean()
    df["swing_high"] = df["high"].rolling(3, center=True).max()
    df["swing_low"] = df["low"].rolling(3, center=True).min()
    return df

def is_bos(df, idx):
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

def detect_order_block(df, idx):
    if idx - 1 < 0:
        return None
    ob_candle = df.iloc[idx-1]
    return (ob_candle["high"], ob_candle["low"])

# ------------------------------
# TRADING LOGIC - FIXED CAPITAL UPDATE
# ------------------------------
def run_strategy():
    global trades_data, summary_data, open_signals, is_running, coin_performance, time_performance
    
    is_running = True
    capital = config["initial_capital"]
    total_withdrawn = 0.0
    total_invested_volume = 0.0
    trades = []
    open_signals = []
    
    # Initialize performance tracking
    coin_performance = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_profit": 0.0, "total_invested": 0.0})
    time_performance = {
        "weekly": defaultdict(lambda: {"trades": 0, "net_profit": 0.0}),
        "monthly": defaultdict(lambda: {"trades": 0, "net_profit": 0.0})
    }
    
    days = config["testing_months"] * 30
    
    for symbol in config["coins"]:
        print(f"\nProcessing {symbol} ...")
        try:
            df = fetch_ohlcv(symbol, days=days)
            df = add_indicators(df)
            daily_trades = {}
            
            for i in range(2, len(df)-1):
                candle = df.iloc[i]
                date = candle.name.date()
                if date not in daily_trades:
                    daily_trades[date] = 0
                if daily_trades[date] >= config["daily_trade_limit_per_coin"]:
                    continue
                
                atr = candle["ATR14"]
                vol_avg = df["VOL_AVG14"].iloc[i]
                if pd.isna(atr) or pd.isna(vol_avg):
                    continue
                
                bos = is_bos(df, i)
                if not bos:
                    continue
                
                if candle["volume"] < config["volume_multiplier"] * vol_avg:
                    continue
                
                trend = "bull" if candle["EMA50"] > candle["EMA200"] else "bear"
                if bos == "bull" and trend != "bull":
                    continue
                if bos == "bear" and trend != "bear":
                    continue
                
                ob = detect_order_block(df, i)
                if ob is None:
                    continue
                ob_high, ob_low = ob
                
                # LONG ENTRY
                if bos == "bull":
                    entry_idx = None
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
                        continue
                    
                    entry_row = df.iloc[entry_idx]
                    entry_price = entry_row["close"]
                    sl_price = min(ob_low - atr * 0.1, entry_price - atr * config["atr_sl_multiplier"])
                    
                    # Fixed Investment Mode
                    if config["fixed_investment_mode"]:
                        invested = config["fixed_investment_amount"]
                        position_size_units = invested / entry_price
                    else:
                        risk_amount = capital * config["risk_per_trade"]
                        risk_per_unit = abs(entry_price - sl_price)
                        if risk_per_unit <= 0:
                            continue
                        
                        position_size_units = risk_amount / risk_per_unit
                        invested = position_size_units * entry_price
                        
                        if invested > capital:
                            scale_factor = capital / invested
                            position_size_units *= scale_factor
                            invested = capital
                    
                    total_invested_volume += invested
                    
                    tp_price = entry_price + (abs(entry_price - sl_price) * config["tp_rr"])
                    result = "OPEN"
                    pnl = 0.0
                    
                    for k in range(entry_idx+1, min(len(df), entry_idx + config["max_lookforward_bars"])):
                        sc = df.iloc[k]
                        if sc["low"] <= sl_price:
                            result = "LOSS"
                            pnl = position_size_units * (sl_price - entry_price)
                            break
                        elif sc["high"] >= tp_price:
                            result = "WIN"
                            pnl = position_size_units * (tp_price - entry_price)
                            break
                    
                    trade_date = entry_row.name
                    trade = {
                        "symbol": symbol,
                        "date": trade_date.strftime("%Y-%m-%d %H:%M"),
                        "type": "LONG",
                        "entry": round(entry_price, 8),
                        "sl": round(sl_price, 8),
                        "tp": round(tp_price, 8),
                        "invested": round(invested, 8),
                        "position_size": round(position_size_units, 8),
                        "result": result,
                        "pnl": round(pnl, 8),
                        "raw_date": trade_date
                    }
                    trades.append(trade)
                    
                    # Update coin performance
                    coin_performance[symbol]["trades"] += 1
                    coin_performance[symbol]["total_invested"] += invested
                    coin_performance[symbol]["net_profit"] += pnl
                    if result == "WIN":
                        coin_performance[symbol]["wins"] += 1
                    elif result == "LOSS":
                        coin_performance[symbol]["losses"] += 1
                    
                    # Update time performance
                    week_key = trade_date.strftime("%Y-W%W")
                    month_key = trade_date.strftime("%Y-%m")
                    
                    time_performance["weekly"][week_key]["trades"] += 1
                    time_performance["weekly"][week_key]["net_profit"] += pnl
                    time_performance["monthly"][month_key]["trades"] += 1
                    time_performance["monthly"][month_key]["net_profit"] += pnl
                    
                    if result == "OPEN":
                        open_signals.append(trade)
                    
                    # ✅ FIXED: CAPITAL UPDATE FOR BOTH MODES
                    if config["fixed_investment_mode"]:
                        # Fixed mode - update capital for tracking profits
                        capital += pnl
                    else:
                        # Normal mode
                        if config["compounding"]:
                            capital += pnl
                        else:
                            capital = config["initial_capital"] + sum(t["pnl"] for t in trades)
                    
                    daily_trades[date] += 1
                
                # SHORT ENTRY
                elif bos == "bear":
                    entry_idx = None
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
                        continue
                    
                    entry_row = df.iloc[entry_idx]
                    entry_price = entry_row["close"]
                    sl_price = max(ob_high + atr * 0.1, entry_price + atr * config["atr_sl_multiplier"])
                    
                    # Fixed Investment Mode
                    if config["fixed_investment_mode"]:
                        invested = config["fixed_investment_amount"]
                        position_size_units = invested / entry_price
                    else:
                        risk_amount = capital * config["risk_per_trade"]
                        risk_per_unit = abs(sl_price - entry_price)
                        if risk_per_unit <= 0:
                            continue
                        
                        position_size_units = risk_amount / risk_per_unit
                        invested = position_size_units * entry_price
                        
                        if invested > capital:
                            scale_factor = capital / invested
                            position_size_units *= scale_factor
                            invested = capital
                    
                    total_invested_volume += invested
                    
                    tp_price = entry_price - (abs(sl_price - entry_price) * config["tp_rr"])
                    result = "OPEN"
                    pnl = 0.0
                    
                    for k in range(entry_idx+1, min(len(df), entry_idx + config["max_lookforward_bars"])):
                        sc = df.iloc[k]
                        if sc["high"] >= sl_price:
                            result = "LOSS"
                            pnl = position_size_units * (entry_price - sl_price)
                            break
                        elif sc["low"] <= tp_price:
                            result = "WIN"
                            pnl = position_size_units * (entry_price - tp_price)
                            break
                    
                    trade_date = entry_row.name
                    trade = {
                        "symbol": symbol,
                        "date": trade_date.strftime("%Y-%m-%d %H:%M"),
                        "type": "SHORT",
                        "entry": round(entry_price, 8),
                        "sl": round(sl_price, 8),
                        "tp": round(tp_price, 8),
                        "invested": round(invested, 8),
                        "position_size": round(position_size_units, 8),
                        "result": result,
                        "pnl": round(pnl, 8),
                        "raw_date": trade_date
                    }
                    trades.append(trade)
                    
                    # Update coin performance
                    coin_performance[symbol]["trades"] += 1
                    coin_performance[symbol]["total_invested"] += invested
                    coin_performance[symbol]["net_profit"] += pnl
                    if result == "WIN":
                        coin_performance[symbol]["wins"] += 1
                    elif result == "LOSS":
                        coin_performance[symbol]["losses"] += 1
                    
                    # Update time performance
                    week_key = trade_date.strftime("%Y-W%W")
                    month_key = trade_date.strftime("%Y-%m")
                    
                    time_performance["weekly"][week_key]["trades"] += 1
                    time_performance["weekly"][week_key]["net_profit"] += pnl
                    time_performance["monthly"][month_key]["trades"] += 1
                    time_performance["monthly"][month_key]["net_profit"] += pnl
                    
                    if result == "OPEN":
                        open_signals.append(trade)
                    
                    # ✅ FIXED: CAPITAL UPDATE FOR BOTH MODES
                    if config["fixed_investment_mode"]:
                        # Fixed mode - update capital for tracking profits
                        capital += pnl
                    else:
                        # Normal mode
                        if config["compounding"]:
                            capital += pnl
                        else:
                            capital = config["initial_capital"] + sum(t["pnl"] for t in trades)
                    
                    daily_trades[date] += 1
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Monthly Withdrawal (Only for normal mode)
    if config["enable_withdrawal"] and not config["fixed_investment_mode"]:
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty:
            df_trades["month"] = pd.to_datetime(df_trades["date"]).dt.to_period("M")
            monthly_profit = df_trades.groupby("month")["pnl"].sum()
            for month, profit in monthly_profit.items():
                if profit > 0:
                    withdraw = profit * config["withdraw_percentage"]
                    reinvest = profit * config["reinvest_percentage"]
                    total_withdrawn += withdraw
                    capital += reinvest
                    capital -= withdraw
    
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    net_profit = (capital + total_withdrawn) - config["initial_capital"]
    
    trades_data = trades
    summary_data = {
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
    
    is_running = False
    print("\n✅ Strategy execution completed!")

# ------------------------------
# FLASK ROUTES
# ------------------------------
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMC Strategy Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1 { font-size: 24px; color: #2c3e50; margin-bottom: 5px; }
        .subtitle { color: #7f8c8d; font-size: 14px; }
        
        .settings-panel {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { 
            font-size: 13px; 
            font-weight: 600; 
            margin-bottom: 5px; 
            color: #555;
        }
        .form-group input, .form-group select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .checkbox-group label {
            font-size: 14px;
            cursor: pointer;
            color: #555;
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .btn:hover { background: #2980b9; }
        .btn:disabled { background: #95a5a6; cursor: not-allowed; }
        .btn-block { width: 100%; }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card h3 { 
            font-size: 13px; 
            color: #7f8c8d; 
            margin-bottom: 8px;
            font-weight: 600;
        }
        .summary-card p { 
            font-size: 24px; 
            font-weight: 700;
            color: #2c3e50;
        }
        .summary-card.profit p { color: #27ae60; }
        .summary-card.loss p { color: #e74c3c; }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            flex: 1;
            min-width: 120px;
            padding: 12px;
            background: white;
            border: 2px solid #e0e0e0;
            color: #555;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.3s;
            text-align: center;
        }
        .tab.active { 
            background: #3498db; 
            color: white;
            border-color: #3498db;
        }
        .tab:hover { border-color: #3498db; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .content-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .content-card h2 { 
            font-size: 18px; 
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        .filter-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-bar select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            background: white;
        }
        .filter-bar label {
            font-size: 13px;
            font-weight: 600;
            color: #555;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
            font-size: 13px;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        tr:hover { background: #f8f9fa; }
        
        .signal-card {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .signal-header { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 10px;
        }
        .signal-type { 
            font-weight: 700; 
            font-size: 16px;
            color: #2c3e50;
        }
        .signal-details { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 10px;
            font-size: 13px;
        }
        .signal-details strong { color: #555; }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge.win { background: #d4edda; color: #155724; }
        .badge.loss { background: #f8d7da; color: #721c24; }
        .badge.open { background: #fff3cd; color: #856404; }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #95a5a6;
        }
        .fixed-investment-panel {
            display: none;
            margin-top: 10px;
        }
        .fixed-investment-panel.active {
            display: block;
        }
        
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .performance-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .performance-section h3 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 SMC Trading Strategy Dashboard</h1>
            <p class="subtitle">Smart Money Concept Backtesting & Signal Generator</p>
        </header>
        
        <div class="settings-panel">
            <h2 style="margin-bottom: 15px; font-size: 16px;">⚙️ Configuration</h2>
            <div class="settings-grid">
                <div class="form-group">
                    <label>Testing Period (Months)</label>
                    <select id="testingMonths">
                        <option value="1">1 Month</option>
                        <option value="3">3 Months</option>
                        <option value="6">6 Months</option>
                        <option value="12">12 Months</option>
                        <option value="24">24 Months</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Initial Capital ($)</label>
                    <input type="number" id="initialCapital" value="50" min="1">
                </div>
                <div class="form-group">
                    <label>Risk Per Trade (%)</label>
                    <input type="number" id="riskPerTrade" value="1" min="0.1" max="100" step="0.1">
                </div>
                <div class="form-group">
                    <label>Daily Trade Limit</label>
                    <input type="number" id="tradeLimit" value="5" min="1">
                </div>
            </div>
            
            <div class="settings-grid">
                <div class="checkbox-group">
                    <input type="checkbox" id="compounding" checked onchange="toggleCompounding()">
                    <label for="compounding">Enable Compounding</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="withdrawal" checked>
                    <label for="withdrawal">Enable Monthly Withdrawal (30%)</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="fixedInvestment" onchange="toggleFixedInvestment()">
                    <label for="fixedInvestment">Fixed Investment Mode</label>
                </div>
            </div>
            
            <div id="fixedInvestmentPanel" class="fixed-investment-panel">
                <div class="form-group" style="max-width: 300px;">
                    <label>Fixed Investment Amount ($)</label>
                    <input type="number" id="fixedAmount" value="10" min="1" step="0.1">
                </div>
            </div>
            
            <button class="btn btn-block" id="runBtn" onclick="runStrategy()">🚀 Run Backtest</button>
        </div>
        
        <div class="summary" id="summary"></div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('performance')">📈 Performance</button>
            <button class="tab" onclick="showTab('signals')">📡 Open Signals</button>
            <button class="tab" onclick="showTab('journal')">📖 Trade Journal</button>
        </div>
        
        <div id="performance" class="tab-content active">
            <div class="performance-grid">
                <div class="performance-section">
                    <h3>💰 Coin-wise Performance</h3>
                    <div id="coinPerformance"></div>
                </div>
                <div class="performance-section">
                    <h3>📅 Weekly Performance</h3>
                    <div id="weeklyPerformance"></div>
                </div>
                <div class="performance-section">
                    <h3>🗓️ Monthly Performance</h3>
                    <div id="monthlyPerformance"></div>
                </div>
            </div>
        </div>
        
        <div id="signals" class="tab-content">
            <div class="content-card">
                <h2>🔔 Active Trading Signals</h2>
                <div class="filter-bar">
                    <label>Filter by Time:</label>
                    <select id="signalFilter" onchange="filterSignals()">
                        <option value="all">All Signals</option>
                        <option value="24h">Last 24 Hours</option>
                        <option value="48h">Last 48 Hours</option>
                        <option value="7d">Last 7 Days</option>
                        <option value="30d">Last 30 Days</option>
                    </select>
                    <label>Coin:</label>
                    <select id="coinFilter" onchange="filterSignals()">
                        <option value="all">All Coins</option>
                    </select>
                </div>
                <div id="signalsContent"></div>
            </div>
        </div>
        
        <div id="journal" class="tab-content">
            <div class="content-card">
                <h2>📊 Complete Trade History</h2>
                <div id="journalContent"></div>
            </div>
        </div>
    </div>
    
    <script>
        let allSignals = [];
        
        function toggleFixedInvestment() {
            const fixedMode = document.getElementById('fixedInvestment').checked;
            const panel = document.getElementById('fixedInvestmentPanel');
            const compounding = document.getElementById('compounding');
            const withdrawal = document.getElementById('withdrawal');
            
            if (fixedMode) {
                panel.classList.add('active');
                compounding.checked = false;
                compounding.disabled = true;
                withdrawal.checked = false;
                withdrawal.disabled = true;
            } else {
                panel.classList.remove('active');
                compounding.disabled = false;
                withdrawal.disabled = false;
            }
        }
        
        function toggleCompounding() {
            const compounding = document.getElementById('compounding').checked;
            const withdrawal = document.getElementById('withdrawal');
            
            if (!compounding) {
                withdrawal.checked = false;
            }
        }
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        function filterSignals() {
            const timeFilter = document.getElementById('signalFilter').value;
            const coinFilter = document.getElementById('coinFilter').value;
            
            let filtered = allSignals;
            
            // Filter by coin
            if (coinFilter !== 'all') {
                filtered = filtered.filter(s => s.symbol === coinFilter);
            }
            
            // Filter by time
            if (timeFilter !== 'all') {
                const now = new Date();
                const hours = {
                    '24h': 24,
                    '48h': 48,
                    '7d': 24 * 7,
                    '30d': 24 * 30
                }[timeFilter];
                
                filtered = filtered.filter(s => {
                    const signalDate = new Date(s.date);
                    const diff = (now - signalDate) / (1000 * 60 * 60);
                    return diff <= hours;
                });
            }
            
            displaySignals(filtered);
        }
        
        function displaySignals(signals) {
            if (signals.length === 0) {
                document.getElementById('signalsContent').innerHTML = '<div class="empty-state"><p>No signals match the filter criteria</p></div>';
            } else {
                document.getElementById('signalsContent').innerHTML = signals.map(s => `
                    <div class="signal-card">
                        <div class="signal-header">
                            <span class="signal-type">${s.type} ${s.symbol}</span>
                            <span style="color: #7f8c8d; font-size: 13px;">${s.date}</span>
                        </div>
                        <div class="signal-details">
                            <div><strong>Entry:</strong> ${s.entry}</div>
                            <div><strong>Stop Loss:</strong> ${s.sl}</div>
                            <div><strong>Take Profit:</strong> ${s.tp}</div>
                            <div><strong>Position Size:</strong> ${s.position_size}</div>
                            <div><strong>Invested:</strong> ${s.invested}</div>
                        </div>
                    </div>
                `).join('');
            }
        }
        
        function displayPerformance(performanceData) {
            // Coin Performance
            let coinHtml = '';
            if (Object.keys(performanceData.coin_performance).length === 0) {
                coinHtml = '<div class="empty-state"><p>No coin performance data</p></div>';
            } else {
                coinHtml = '<table><thead><tr><th>Coin</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Net Profit</th></tr></thead><tbody>';
                for (const [coin, data] of Object.entries(performanceData.coin_performance)) {
                    const winRate = data.trades > 0 ? ((data.wins / data.trades) * 100).toFixed(2) : '0.00';
                    coinHtml += `
                        <tr>
                            <td><strong>${coin}</strong></td>
                            <td>${data.trades}</td>
                            <td style="color: #27ae60;">${data.wins}</td>
                            <td style="color: #e74c3c;">${data.losses}</td>
                            <td>${winRate}%</td>
                            <td style="color: ${data.net_profit >= 0 ? '#27ae60' : '#e74c3c'}; font-weight: 600;">${data.net_profit.toFixed(2)}</td>
                        </tr>
                    `;
                }
                coinHtml += '</tbody></table>';
            }
            document.getElementById('coinPerformance').innerHTML = coinHtml;
            
            // Weekly Performance
            let weeklyHtml = '';
            if (Object.keys(performanceData.weekly_performance).length === 0) {
                weeklyHtml = '<div class="empty-state"><p>No weekly performance data</p></div>';
            } else {
                weeklyHtml = '<table><thead><tr><th>Week</th><th>Trades</th><th>Net Profit</th></tr></thead><tbody>';
                for (const [week, data] of Object.entries(performanceData.weekly_performance)) {
                    weeklyHtml += `
                        <tr>
                            <td><strong>${week}</strong></td>
                            <td>${data.trades}</td>
                            <td style="color: ${data.net_profit >= 0 ? '#27ae60' : '#e74c3c'}; font-weight: 600;">${data.net_profit.toFixed(2)}</td>
                        </tr>
                    `;
                }
                weeklyHtml += '</tbody></table>';
            }
            document.getElementById('weeklyPerformance').innerHTML = weeklyHtml;
            
            // Monthly Performance
            let monthlyHtml = '';
            if (Object.keys(performanceData.monthly_performance).length === 0) {
                monthlyHtml = '<div class="empty-state"><p>No monthly performance data</p></div>';
            } else {
                monthlyHtml = '<table><thead><tr><th>Month</th><th>Trades</th><th>Net Profit</th></tr></thead><tbody>';
                for (const [month, data] of Object.entries(performanceData.monthly_performance)) {
                    monthlyHtml += `
                        <tr>
                            <td><strong>${month}</strong></td>
                            <td>${data.trades}</td>
                            <td style="color: ${data.net_profit >= 0 ? '#27ae60' : '#e74c3c'}; font-weight: 600;">${data.net_profit.toFixed(2)}</td>
                        </tr>
                    `;
                }
                monthlyHtml += '</tbody></table>';
            }
            document.getElementById('monthlyPerformance').innerHTML = monthlyHtml;
        }
        
        function runStrategy() {
            const settings = {
                testing_months: parseInt(document.getElementById('testingMonths').value),
                initial_capital: parseFloat(document.getElementById('initialCapital').value),
                risk_per_trade: parseFloat(document.getElementById('riskPerTrade').value) / 100,
                daily_trade_limit_per_coin: parseInt(document.getElementById('tradeLimit').value),
                compounding: document.getElementById('compounding').checked,
                enable_withdrawal: document.getElementById('withdrawal').checked,
                fixed_investment_mode: document.getElementById('fixedInvestment').checked,
                fixed_investment_amount: parseFloat(document.getElementById('fixedAmount').value)
            };
            
            document.getElementById('runBtn').disabled = true;
            document.getElementById('runBtn').textContent = '⏳ Running Backtest...';
            document.getElementById('summary').innerHTML = '<div class="loading">🔄 Processing historical data...</div>';
            
            fetch('/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            }).then(r => r.json()).then(data => {
                setTimeout(loadData, 3000);
            });
        }
        
        function loadData() {
            fetch('/data').then(r => r.json()).then(data => {
                if (data.is_running) {
                    setTimeout(loadData, 2000);
                    return;
                }
                
                document.getElementById('runBtn').disabled = false;
                document.getElementById('runBtn').textContent = '🚀 Run Backtest';
                
                // Summary
                let s = data.summary;
                if (Object.keys(s).length === 0) {
                    document.getElementById('summary').innerHTML = '<div class="loading">No data yet. Run the backtest to see results.</div>';
                } else {
                    document.getElementById('summary').innerHTML = `
                        <div class="summary-card">
                            <h3>Initial Capital</h3>
                            <p>${s.initial_capital}</p>
                        </div>
                        <div class="summary-card ${s.net_profit >= 0 ? 'profit' : 'loss'}">
                            <h3>Ending Capital</h3>
                            <p>${s.ending_capital}</p>
                        </div>
                        <div class="summary-card">
                            <h3>Total Withdrawn</h3>
                            <p>${s.total_withdrawn}</p>
                        </div>
                        <div class="summary-card">
                            <h3>Total Invested Volume</h3>
                            <p>${s.total_invested_volume}</p>
                        </div>
                        <div class="summary-card">
                            <h3>Total Trades</h3>
                            <p>${s.total_trades}</p>
                        </div>
                        <div class="summary-card">
                            <h3>Win Rate</h3>
                            <p>${s.winrate}%</p>
                        </div>
                        <div class="summary-card ${s.net_profit >= 0 ? 'profit' : 'loss'}">
                            <h3>Net Profit</h3>
                            <p>${s.net_profit}</p>
                        </div>
                        <div class="summary-card profit">
                            <h3>Wins</h3>
                            <p>${s.wins}</p>
                        </div>
                        <div class="summary-card loss">
                            <h3>Losses</h3>
                            <p>${s.losses}</p>
                        </div>
                    `;
                }
                
                // Display performance data
                displayPerformance(data.performance);
                
                // Store signals and update coin filter
                allSignals = data.signals;
                const uniqueCoins = [...new Set(allSignals.map(s => s.symbol))];
                const coinFilter = document.getElementById('coinFilter');
                coinFilter.innerHTML = '<option value="all">All Coins</option>' + 
                    uniqueCoins.map(coin => `<option value="${coin}">${coin}</option>`).join('');
                
                // Display filtered signals
                filterSignals();
                
                // Journal
                if (data.trades.length === 0) {
                    document.getElementById('journalContent').innerHTML = '<div class="empty-state"><p>No trades recorded yet</p></div>';
                } else {
                    document.getElementById('journalContent').innerHTML = `
                        <div style="overflow-x: auto;">
                            <table>
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Symbol</th>
                                        <th>Date</th>
                                        <th>Type</th>
                                        <th>Entry</th>
                                        <th>SL</th>
                                        <th>TP</th>
                                        <th>Invested</th>
                                        <th>Result</th>
                                        <th>PnL</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.trades.map((t, i) => `
                                        <tr>
                                            <td>${i+1}</td>
                                            <td><strong>${t.symbol}</strong></td>
                                            <td>${t.date}</td>
                                            <td>${t.type}</td>
                                            <td>${t.entry}</td>
                                            <td>${t.sl}</td>
                                            <td>${t.tp}</td>
                                            <td>${t.invested}</td>
                                            <td><span class="badge ${t.result.toLowerCase()}">${t.result}</span></td>
                                            <td style="color: ${t.pnl >= 0 ? '#27ae60' : '#e74c3c'}; font-weight: 600;">${t.pnl}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                }
            });
        }
        
        // Load data on page load
        loadData();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/run', methods=['POST'])
def run():
    global config
    if not is_running:
        settings = request.get_json()
        config.update(settings)
        threading.Thread(target=run_strategy, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/data')
def data():
    return jsonify({
        "trades": trades_data,
        "summary": summary_data,
        "signals": open_signals,
        "performance": {
            "coin_performance": dict(coin_performance),
            "weekly_performance": dict(time_performance["weekly"]),
            "monthly_performance": dict(time_performance["monthly"])
        },
        "is_running": is_running
    })

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    print("🚀 Starting SMC Strategy Dashboard...")
    print("📡 Access at: http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)