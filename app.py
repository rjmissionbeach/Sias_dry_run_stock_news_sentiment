import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import json
from datetime import date, timedelta, datetime

FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

model = "nvidia/nemotron-3-super-120b-a12b:free"


def score_sentiment(headline, summary, ticker):
    prompt = f"""
    Score the sentiment of this stock-news article for {ticker}.

    Return only JSON with:
    - score: a number from -1 to +1
    - label: positive, neutral, or negative

    Headline: {headline}
    Summary: {summary}
    """

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


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

if news:
    news_df = pd.DataFrame(news)

    news_df["date"] = news_df["datetime"].apply(
        lambda x: datetime.fromtimestamp(x).date()
    )

    selected_parts = []

    for _, group in news_df.groupby("date"):
        n = min(len(group), 3)
        selected_parts.append(
            group.sample(n=n, random_state=42)
        )

    selected_df = (
        pd.concat(selected_parts)
        .sort_values("date")
        .head(article_limit)
    )

        if st.button("Analyze sentiment"):
        results = []

        for _, article in selected_df.iterrows():
            sentiment = score_sentiment(
                article["headline"],
                article["summary"],
                ticker
            )

            results.append({
                "date": article["date"],
                "headline": article["headline"],
                "source": article["source"],
                "score": sentiment["score"],
                "label": sentiment["label"]
            })

        sentiment_df = pd.DataFrame(results)

        st.write("Articles scored:", len(sentiment_df))
        st.dataframe(sentiment_df.head())

    st.write("Articles selected:", len(selected_df))
    st.write("News dates represented:", selected_df["date"].nunique())
