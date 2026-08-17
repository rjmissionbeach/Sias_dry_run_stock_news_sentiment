import streamlit as st

st.title("Stock News Sentiment Analyzer")

ticker = st.text_input("Enter a ticker:", value="AAPL")

if ticker:
    st.write("Ticker entered:", ticker)
