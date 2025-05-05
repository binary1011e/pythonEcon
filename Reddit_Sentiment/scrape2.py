#!/usr/bin/env python3
import io
import json
import csv
from pathlib import Path
import zstandard as zstd
from datetime import datetime


# This is to get the comments of the posts that have "naitonal debt" or "national deficit" in them
# ── CONFIG ───────────────────────────────────────────────────────────────
SUB_ZST    = "Economics_submissions.zst"    # your raw submissions dump
CMT_ZST    = "Economics_comments.zst"       # your raw comments dump
OUT_IDS    = "economics_submission_ids.txt"
OUT_CMTS   = "filtered_economics_comments.csv"

KEYWORDS   = ["national debt", "national deficit"]

# ── 1) EXTRACT matching submission IDs ────────────────────────────────────
subs_stream = Path(SUB_ZST).open("rb")
subs_reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(subs_stream)
with open(OUT_IDS, "w", encoding="utf-8") as f_ids:
    for line in io.TextIOWrapper(subs_reader, encoding="utf-8"):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = obj.get("title", "") or ""
        selftext = obj.get("selftext", "") or ""
        text = f"{title} {selftext}".lower()
        if any(k in text for k in KEYWORDS):
            f_ids.write(obj["id"] + "\n")
subs_stream.close()

# load IDs into a set for fast lookups
with open(OUT_IDS, "r", encoding="utf-8") as f:
    keep_ids = set(line.strip() for line in f)

# ── 2) FILTER comments & WRITE CSV ────────────────────────────────────────
cmts_stream = Path(CMT_ZST).open("rb")
cmts_reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(cmts_stream)
with open(OUT_CMTS, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
    # write header
    writer.writerow([
        "id","created_utc","subreddit","author",
        "score","num_comments","type","text","permalink"
    ])
    # iterate comments
    for line in io.TextIOWrapper(cmts_reader, encoding="utf-8"):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # check if parent submission matches
        parent = obj.get("link_id","").split("_",1)[-1]
        if parent not in keep_ids:
            continue
        # extract fields
        cid          = obj.get("id", "")
        created_ts   = int(obj.get("created_utc", 0))
        created_iso  = datetime.utcfromtimestamp(created_ts).isoformat()
        subreddit    = obj.get("subreddit", "")
        author       = obj.get("author", "")
        score        = obj.get("score", 0)
        num_comments = obj.get("num_comments", 0)
        text         = obj.get("body", "").replace("\n", " ")
        permalink    = obj.get("permalink", "")
        # write row
        writer.writerow([
            cid, created_iso, subreddit, author,
            score, num_comments, "comment", text, permalink
        ])
cmts_stream.close()

