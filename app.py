import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

st.title("Stock News Sentiment Analyzer")

ticker = st.text_input("Enter a ticker:", value="AAPL").upper()

article_limit = st.slider(
    "Number of articles to analyze",
    min_value=5,
    max_value=50,
    value=25,
    step=5
)

end_date = date.today()
start_date = end_date - timedelta(days=30)

url = "https://finnhub.io/api/v1/company-news"

params = {
    "symbol": ticker,
    "from": start_date.isoformat(),
    "to": end_date.isoformat()
}

headers = {
    "X-Finnhub-Token": FINNHUB_API_KEY
}

response = requests.get(url, params=params, headers=headers)
news = response.json()

st.write("Finnhub status code:", response.status_code)
st.write("Articles returned:", len(news))

st.write("Finnhub key loaded:", bool(FINNHUB_API_KEY))
st.write("Ticker entered:", ticker)
st.write("Articles requested:", article_limit)
st.write("OpenRouter key loaded:", bool(OPENROUTER_API_KEY))
