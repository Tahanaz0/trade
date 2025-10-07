import streamlit as st
import requests
import pandas as pd

# Streamlit page settings
st.set_page_config(page_title="Crypto Data Viewer", page_icon="📊")

# Title
st.title("📈 Crypto Market Live Data (CoinGecko)")

# Popular coins ka list
coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "BNB": "binancecoin",
    "Solana": "solana",
    "Cardano": "cardano",
    "XRP": "ripple",
    "Polygon": "matic-network"
}

# User select karega
coin_name = st.selectbox("Select Coin:", list(coins.keys()))
coin_id = coins[coin_name]
currency = st.selectbox("Select currency:", ["usd", "eur", "pkr"])
days = st.slider("Number of days:", 1, 30, 7)

try:
    # API call without interval
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": currency, "days": days}
    r = requests.get(url, params=params)
    data = r.json()

    if "prices" in data and len(data["prices"]) > 0:
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["Time", "Price"])
        df["Time"] = pd.to_datetime(df["Time"], unit="ms")

        # Show latest data
        st.subheader(f"{coin_name} Price in {currency.upper()}")
        st.dataframe(df.tail())

        # Chart
        st.subheader("📉 Price Chart")
        st.line_chart(df.set_index("Time")["Price"])
    else:
        st.error("⚠️ No data available. Try reducing 'days' or select another coin.")

except Exception as e:
    st.error(f"Error: {str(e)}")
