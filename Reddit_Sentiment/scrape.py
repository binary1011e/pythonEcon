#!/usr/bin/env python3
import io
import json
import csv
from datetime import datetime
import zstandard as zstd


# This is to get the comments or the posts that have "naitonal debt" or "national deficit" in them
# ── CONFIG ───────────────────────────────────────────────────────────────
INPUT_ZST   = "Economics_comments.zst"   # path to your .zst file (Not public; I am not putting 26 GB on Github)
OUTPUT_CSV  = "Economics_debt_deficit_comments.csv"        # your output CSV

# terms to search for (case‐insensitive substring match)
TERMS = ["national debt", "national deficit"]

# how many characters of text context to keep (None = whole field)
MAX_TEXT_LEN = 500

# ── STREAM & FILTER ──────────────────────────────────────────────────────
def stream_and_filter(input_zst, output_csv):
    # open output CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        # header row
        writer.writerow([
            "id", "created_utc", "subreddit", "author",
            "score", "num_comments",
            "type",   # "submission" or "comment"
            "title",  # submissions only
            "text",   # selftext or comment body
            "permalink"
        ])

        # set up Zstd decompressor + text wrapper
        with open(input_zst, "rb") as ifh:
            dctx   = zstd.ZstdDecompressor(max_window_size=2**31)
            reader = dctx.stream_reader(ifh)
            text_f = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # build one long string to search
                title    = obj.get("title", "")
                selftext = obj.get("selftext", "")    # submissions
                body     = obj.get("body", "")        # comments
                haystack = f"{title} {selftext} {body}".lower()

                if any(term in haystack for term in TERMS):
                    # determine if submission or comment
                    is_sub = "selftext" in obj and "title" in obj
                    post_type = "submission" if is_sub else "comment"

                    # extract a reasonable snippet
                    text_field = selftext if is_sub else body
                    snippet    = (text_field[:MAX_TEXT_LEN] + "...") if MAX_TEXT_LEN and len(text_field)>MAX_TEXT_LEN else text_field

                    # permalink: either /r/.../comments/... or comment permalink
                    permalink = obj.get("permalink") or obj.get("url","")

                    # write row
                    writer.writerow([
                        obj.get("id",""),
                        datetime.utcfromtimestamp(int(obj["created_utc"])).isoformat(),
                        obj.get("subreddit",""),
                        obj.get("author",""),
                        obj.get("score",0),
                        obj.get("num_comments",0),
                        post_type,
                        title if is_sub else "",
                        snippet,
                        permalink
                    ])

    print(f"Done—filtered rows written to {output_csv}")

if __name__ == "__main__":
    stream_and_filter(INPUT_ZST, OUTPUT_CSV)
