import csv

import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIGURE ───────────────────────────────────────────────────────────────
FILEPATH = "RawRedditCount/Economics_debt_deficit_posts.csv"  # path to your CSV file

# ── 1) LOAD & PARSE ─────────────────────────────────────────────────────────
df = pd.read_csv(FILEPATH, parse_dates=["created_utc"])
print(df.iloc[0])
# ── 2) EXTRACT MONTH ────────────────────────────────────────────────────────
df["month"] = df["created_utc"].dt.to_period("M").dt.to_timestamp()

# ── 3) AGGREGATE BY MONTH ───────────────────────────────────────────────────
monthly = df.groupby("month").agg(
    submission_count=("id", "count"),
    total_score=("score", "sum")
).reset_index()

# ── 4) SMOOTHING (3‐month moving average) ───────────────────────────────────
window = 3
monthly["count_smooth"] = monthly["submission_count"].rolling(window, center=True, min_periods=1).mean()
monthly["score_smooth"] = monthly["total_score"].rolling(window, center=True, min_periods=1).mean()

# ── 5) PLOT MONTHLY SUBMISSION COUNT & SMOOTHED ────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(monthly["month"], monthly["submission_count"], marker="o", linewidth=1.2, alpha=0.6, label="Monthly Count")
plt.plot(monthly["month"], monthly["count_smooth"], color="C2", linewidth=2, label=f"{window}-Month MA")
plt.title("Monthly Debt Comment Count on Debt Posts with 3-Month Moving Average on r/Politics")
plt.xlabel("Month")
plt.ylabel("Number of Posts")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 6) PLOT MONTHLY TOTAL SCORE & SMOOTHED ─────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(monthly["month"], monthly["total_score"], marker="o", linewidth=1.2, alpha=0.6, color="C1", label="Monthly Total Score")
plt.plot(monthly["month"], monthly["score_smooth"], color="C3", linewidth=2, label=f"{window}-Month MA")
plt.title("Monthly Total Score of Posts with 3-Month Moving Average on r/Politics")
plt.xlabel("Month")
plt.ylabel("Total Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
