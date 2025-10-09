import ccxt
import pandas as pd
import pytz
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            print("✅ Binance exchange connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Binance: {e}")
            self.exchange = None
    
    def fetch_ohlcv(self, symbol, timeframe, days=30):
        """Fetch OHLCV data for a symbol"""
        if not self.exchange:
            print("❌ Exchange not initialized")
            return pd.DataFrame()
            
        try:
            since = self.exchange.parse8601(
                (datetime.now().astimezone(pytz.UTC) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            all_ohlcv = []
            limit = 1000
            
            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv += ohlcv
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < limit:
                    break
            
            if not all_ohlcv:
                print(f"❌ No OHLCV data found for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Karachi")
            df.set_index("timestamp", inplace=True)
            
            print(f"✅ Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_indicators(self, df):
        """Add technical indicators to DataFrame"""
        if df.empty:
            return df
            
        try:
            df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
            
            # RSI
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))
            
            # ATR
            df["H-L"] = df["high"] - df["low"]
            df["H-PC"] = (df["high"] - df["close"].shift(1)).abs()
            df["L-PC"] = (df["low"] - df["close"].shift(1)).abs()
            df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            df["ATR14"] = df["TR"].rolling(14).mean()
            
            # Volume
            df["VOL_AVG14"] = df["volume"].rolling(14).mean()
            
            # Swing points
            df["swing_high"] = df["high"].rolling(3, center=True).max()
            df["swing_low"] = df["low"].rolling(3, center=True).min()
            
            print("✅ Indicators added successfully")
            return df
            
        except Exception as e:
            print(f"❌ Error adding indicators: {e}")
            return df