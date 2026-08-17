import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]

st.title("Stock News Sentiment Analyzer")

ticker = st.text_input("Enter a ticker:", value="AAPL").upper()

article_limit = st.slider(
    "Number of articles to analyze",
    min_value=5,
    max_value=50,
    value=25,
    step=5
)

st.write("Finnhub key loaded:", bool(FINNHUB_API_KEY))
st.write("Ticker entered:", ticker)
st.write("Articles requested:", article_limit)
