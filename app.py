import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
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

    data = response.json()

    if response.status_code != 200 or "choices" not in data:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: {data}"
        )

    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


st.title("Stock News Sentiment Analyzer")

ticker = st.text_input(
    "Enter a ticker:",
    value="AAPL"
).upper()

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

response = requests.get(
    url,
    params=params,
    headers=headers
)

news = response.json()

if response.status_code != 200:
    st.error("Finnhub news request failed.")
    st.stop()

if not news:
    st.warning(
        f"No recent Finnhub news was found for {ticker}."
    )
    st.stop()


news_df = pd.DataFrame(news)

news_df["date"] = news_df["datetime"].apply(
    lambda x: datetime.fromtimestamp(x).date()
)

selected_parts = []

for _, group in news_df.groupby("date"):
    n = min(len(group), 3)

    selected_parts.append(
        group.sample(
            n=n,
            random_state=42
        )
    )

selected_df = (
    pd.concat(selected_parts)
    .sort_values("date")
    .head(article_limit)
)

st.write(
    f"Finnhub returned **{len(news)} articles** "
    f"from **{news_df['date'].min()}** "
    f"through **{news_df['date'].max()}**."
)

st.write(
    f"Analyzing **{len(selected_df)} articles** "
    f"across **{selected_df['date'].nunique()} news dates**."
)


if st.button("Analyze sentiment"):

    results = []

    with st.spinner("Scoring news sentiment..."):

        for _, article in selected_df.iterrows():

            try:
                sentiment = score_sentiment(
                    article["headline"],
                    article["summary"],
                    ticker
                )

                results.append({
                    "date": article["date"],
                    "headline": article["headline"],
                    "source": article["source"],
                    "url": article["url"],
                    "score": float(sentiment["score"]),
                    "label": sentiment["label"].lower()
                })

            except Exception as e:
                st.warning(
                    f"One article could not be scored: {e}"
                )

    sentiment_df = pd.DataFrame(results)

    if sentiment_df.empty:
        st.error("No articles could be scored.")
        st.stop()

    # -----------------------------
    # Sentiment distribution
    # -----------------------------

    sentiment_counts = (
        sentiment_df["label"]
        .value_counts()
        .reindex(
            ["negative", "neutral", "positive"],
            fill_value=0
        )
    )

    st.subheader("Sentiment Distribution")

    st.bar_chart(sentiment_counts)

    # -----------------------------
    # Price data
    # -----------------------------

    price_df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True
    )

    if price_df.empty:
        st.error("No price data could be retrieved.")
        st.stop()

    price_clean = price_df["Close"].reset_index()
    price_clean.columns = ["date", "close"]

    price_clean["date"] = (
        pd.to_datetime(price_clean["date"]).dt.date
    )

    price_clean["next_close"] = (
        price_clean["close"].shift(-1)
    )

    # -----------------------------
    # Daily sentiment
    # -----------------------------

    daily_sentiment = (
        sentiment_df
        .groupby("date", as_index=False)
        .agg(
            avg_sentiment=("score", "mean"),
            article_count=("score", "size")
        )
    )

    trading_dates = sorted(price_clean["date"])

    def next_trading_day(news_date):
        for trading_date in trading_dates:
            if trading_date >= news_date:
                return trading_date
        return None

    daily_sentiment["aligned_date"] = (
        daily_sentiment["date"].apply(
            next_trading_day
        )
    )

    daily_sentiment = daily_sentiment.dropna(
        subset=["aligned_date"]
    )

    daily_sentiment["weighted_sentiment"] = (
        daily_sentiment["avg_sentiment"]
        * daily_sentiment["article_count"]
    )

    aligned_sentiment = (
        daily_sentiment
        .groupby("aligned_date", as_index=False)
        .agg(
            weighted_sum=("weighted_sentiment", "sum"),
            article_count=("article_count", "sum")
        )
    )

    aligned_sentiment["avg_sentiment"] = (
        aligned_sentiment["weighted_sum"]
        / aligned_sentiment["article_count"]
    )

    # -----------------------------
    # Hit rate
    # -----------------------------

    evaluation = aligned_sentiment.merge(
        price_clean,
        left_on="aligned_date",
        right_on="date",
        how="left"
    )

    evaluation["forward_return"] = (
        evaluation["next_close"]
        / evaluation["close"]
        - 1
    )

    evaluation["hit"] = pd.NA

    valid = (
        evaluation["forward_return"].notna()
        & (evaluation["avg_sentiment"] != 0)
    )

    evaluation.loc[valid, "hit"] = (
        (
            (evaluation.loc[valid, "avg_sentiment"] > 0)
            &
            (evaluation.loc[valid, "forward_return"] > 0)
        )
        |
        (
            (evaluation.loc[valid, "avg_sentiment"] < 0)
            &
            (evaluation.loc[valid, "forward_return"] < 0)
        )
    )

    evaluable = evaluation[
        evaluation["hit"].notna()
    ].copy()

    st.subheader("Directional Hit Rate")

    if len(evaluable) > 0:

        hit_rate = evaluable["hit"].mean()

        st.metric(
            "Sentiment Direction Hit Rate",
            f"{hit_rate:.1%}"
        )

        st.caption(
            f"Based on {len(evaluable)} evaluable "
            f"trading-day sentiment signal(s). "
            f"Neutral signals and days without a "
            f"subsequent closing price are excluded."
        )

    else:
        st.warning(
            "There are not enough evaluable "
            "directional signals to calculate a hit rate."
        )

    # -----------------------------
    # Price + sentiment chart
    # -----------------------------

    timeline = price_clean.merge(
        aligned_sentiment[
            ["aligned_date", "avg_sentiment"]
        ],
        left_on="date",
        right_on="aligned_date",
        how="left"
    )

    st.subheader("Price and Sentiment Timeline")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        timeline["date"],
        timeline["close"],
        linewidth=2
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Closing Price")
    ax1.tick_params(
        axis="x",
        rotation=45
    )

    ax2 = ax1.twinx()

    ax2.plot(
        timeline["date"],
        timeline["avg_sentiment"],
        marker="o",
        linestyle="--",
        linewidth=2
    )

    ax2.axhline(
        0,
        linewidth=1,
        linestyle=":"
    )

    ax2.set_ylabel("Average Sentiment")
    ax2.set_ylim(-1, 1)

    plt.title(
        f"{ticker}: Price and News Sentiment"
    )

    fig.tight_layout()

    st.pyplot(fig)

    # -----------------------------
    # Article list
    # -----------------------------

    st.subheader("Analyzed News Articles")

    article_display = sentiment_df[
        [
            "date",
            "headline",
            "source",
            "score",
            "label"
        ]
    ].sort_values(
        "date",
        ascending=False
    )

    st.dataframe(
        article_display,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Limitations
    # -----------------------------

    st.caption(
        "Sentiment scores are AI-generated judgments, "
        "not calibrated probabilities. News coverage may "
        "not span the entire requested 30-day window, and "
        "the directional hit rate can be based on a small "
        "number of trading-day observations."
    )
