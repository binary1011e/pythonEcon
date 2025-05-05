import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm
import csv
# ── CONFIG ───────────────────────────────────────────────────────────────
INPUT_CSV  = "economics_debt_deficit_posts.csv"        # your submissions/comments CSV
OUTPUT_CSV = "submission_economics_sentiment.csv" # where to save model outputs

# ── 1) LOAD ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ── 2) BUILD THE “analyze_text” COLUMN ────────────────────────────────────
#   - if type=="submission", take only the title
#   - otherwise (comments), take the text field
if "type" not in df.columns:
    raise ValueError("CSV must have a 'type' column with values 'submission' or 'comment'")
if "title" not in df.columns and "text" not in df.columns:
    raise ValueError("CSV must have both 'title' and 'text' columns")

df["analyze_text"] = df.apply(
    lambda row: row["title"]
                if row["type"].strip().lower() == "submission"
                else row["text"],
    axis=1
).fillna("")

# drop rows where there's nothing to analyze (e.g. “[deleted]” comments)
df = df[df["analyze_text"].str.strip().str.lower() != "[deleted]"].copy()

# ── 3) TWITTER RoBERTa SENTIMENT ─────────────────────────────────────────
twitter_roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    top_k=None,   # return all scores so we can pick the best
    device=0      # set to -1 for CPU-only
)

def twitter_label(text: str) -> str:
    scores = twitter_roberta(text[:512])[0]
    # labels: 'LABEL_0' = negative, 'LABEL_1' = neutral, 'LABEL_2' = positive
    best = max(scores, key=lambda d: d["score"])["label"]
    return {"LABEL_0": "neg", "LABEL_1": "neu", "LABEL_2": "pos"}[best]

# apply with progress bar
df["twitter_label"] = [
    twitter_label(t) for t in tqdm(df["analyze_text"], desc="Twitter-RoBERTa")
]

# ── 4) SAVE OUTPUTS ───────────────────────────────────────────────────────
df.to_csv(
    OUTPUT_CSV,
    index=False,
    quoting=csv.QUOTE_ALL,    # wrap every field in quotes
    quotechar='"'
)
print(f"Saved Twitter-RoBERTa sentiment for {len(df)} items to '{OUTPUT_CSV}'")
