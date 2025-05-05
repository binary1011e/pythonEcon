import pandas as pd
import matplotlib.pyplot as plt

# 1) load your sentiment‐annotated file
df = pd.read_csv(
    "SentimentData/comment_economics_sentiment.csv",
    engine="python",
    parse_dates=["created_utc"],
    on_bad_lines="skip"     # drop any malformed rows
)

# 2) extract month
df["month"] = df["created_utc"].dt.to_period("M").dt.to_timestamp()

# 3) group & count
sentiment_counts = (
    df
    .groupby(["month", "twitter_label"])
    .size()
    .unstack(fill_value=0)   # columns: neg, neu, pos
)

# make sure we have the columns in the right order
for col in ["pos","neu","neg"]:
    if col not in sentiment_counts:
        sentiment_counts[col] = 0
sentiment_counts = sentiment_counts[["pos","neu","neg"]]

# 4) plot counts
plt.figure(figsize=(12,6))
for label, color in [("pos","C2"), ("neu","C1"), ("neg","C0")]:
    plt.plot(
        sentiment_counts.index,
        sentiment_counts[label],
        label=label.capitalize(),
        color=color, marker="o", lw=1.5
    )

plt.title("Monthly Sentiment for r/Economics Comments (Twitter-RoBERTa)")
plt.xlabel("Month")
plt.ylabel("Number of Comments")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5) compute net sentiment = (Pos - Neg) / Total
total = sentiment_counts.sum(axis=1)
net = (sentiment_counts["pos"] - sentiment_counts["neg"]) / total

# 6) plot net sentiment
plt.figure(figsize=(12,4))
plt.plot(
    net.index, net, label="Net Sentiment", color="C3", marker="o", lw=1.5
)
plt.axhline(0, color="gray", ls="--", lw=1)
plt.title("Monthly Net Sentiment for r/Politics Posts ((Pos - Neg) / Total)")
plt.xlabel("Month")
plt.ylabel("Net Sentiment")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
