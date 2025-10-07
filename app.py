import streamlit as st
import ccxt
import pandas as pd

# Streamlit page settings
st.set_page_config(page_title="Crypto Data Viewer", page_icon="📊")

# Title
st.title("📈 Crypto Market Live Data")

# User se symbol choose karne ka option
symbol = st.text_input("Enter symbol (e.g. BTC/USDT):", "BTC/USDT")
timeframe = st.selectbox("Select timeframe:", ["1m", "5m", "15m", "1h", "4h", "1d"])
limit = st.slider("Number of candles:", 5, 200, 20)

# Exchange init
exchange = ccxt.binance()

try:
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # DataFrame me convert
    df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "Volume"])

    # Time ko readable banate hain
    df["Time"] = pd.to_datetime(df["Time"], unit="ms")

    # Show data table
    st.subheader(f"Latest {limit} candles for {symbol}")
    st.dataframe(df)

    # Line chart of Close prices
    st.subheader("📉 Closing Price Chart")
    st.line_chart(df.set_index("Time")["Close"])

except Exception as e:
    st.error(f"Error: {str(e)}")
