"""Lightweight inspection of the raw Customer Support on Twitter CSV.

Reads data/raw/twcs.csv and prints schema, row counts, missingness,
`inbound` distribution, timestamp range, and relationship-field stats.
Does not select a brand, define intents, or build any model/agent code.

Usage:
    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "twcs.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"Dataset not found at {RAW_PATH}")

    section("FILE INFO")
    size_bytes = RAW_PATH.stat().st_size
    print(f"path: {RAW_PATH}")
    print(f"size_bytes: {size_bytes:,} ({size_bytes / (1024 ** 2):.1f} MB)")

    # dtype=str to avoid pandas guessing types on IDs (they mix numeric author_ids
    # with string handles like 'sprintcare'); parse timestamps separately.
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])

    section("ROW COUNT")
    print(f"rows: {len(df):,}")
    print(f"columns: {list(df.columns)}")

    section("SCHEMA / SAMPLE ROWS")
    print(df.head(5).to_string())

    section("MISSING VALUE COUNTS (per column)")
    print(df.isna().sum().to_string())

    section("INBOUND DISTRIBUTION")
    print(df["inbound"].value_counts(dropna=False).to_string())

    section("CREATED_AT RANGE")
    parsed = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    n_unparsed = parsed.isna().sum() - df["created_at"].isna().sum()
    print(f"earliest: {parsed.min()}")
    print(f"latest: {parsed.max()}")
    print(f"rows that failed to parse as datetime (excluding originally missing): {n_unparsed}")

    section("TEXT FIELD")
    empty_text = (df["text"].isna() | (df["text"].str.strip() == "")).sum()
    print(f"missing/empty text rows: {empty_text}")
    print("sample texts:")
    for t in df["text"].dropna().head(3):
        print(f"  - {t[:120]!r}")

    section("RELATIONSHIP FIELDS")
    for col in ["response_tweet_id", "in_response_to_tweet_id"]:
        populated = df[col].notna().sum()
        print(f"{col}: populated={populated:,} ({populated / len(df):.1%}), missing={len(df) - populated:,}")

    # response_tweet_id can hold comma-separated lists of multiple child tweet ids.
    multi = df["response_tweet_id"].dropna().str.contains(",").sum()
    print(f"response_tweet_id rows with multiple (comma-separated) ids: {multi:,}")

    section("AUTHOR_ID / INBOUND CROSS-CHECK (company account identification)")
    inbound_true_authors = df.loc[df["inbound"] == "True", "author_id"].nunique()
    inbound_false_authors = df.loc[df["inbound"] == "False", "author_id"].nunique()
    print(f"distinct author_id values where inbound=True (customers): {inbound_true_authors:,}")
    print(f"distinct author_id values where inbound=False (support accounts): {inbound_false_authors:,}")

    false_authors = sorted(df.loc[df["inbound"] == "False", "author_id"].unique())
    print(f"all inbound=False author_id values ({len(false_authors)} total):")
    print(", ".join(false_authors))

    # Check whether any author_id appears on both sides (would complicate the
    # "inbound=False author_id == brand handle" assumption).
    true_set = set(df.loc[df["inbound"] == "True", "author_id"].unique())
    false_set = set(false_authors)
    overlap = true_set & false_set
    print(f"author_id values appearing as BOTH inbound=True and inbound=False: {len(overlap)}")
    if overlap:
        print(f"  overlap values: {sorted(overlap)[:20]}")

    section("EXPLICIT BRAND/COMPANY COLUMN")
    print("No column named 'brand', 'company', or similar exists in the schema.")
    print(f"Actual columns: {list(df.columns)}")

    section("TWEET_ID UNIQUENESS")
    dup_count = df["tweet_id"].duplicated().sum()
    print(f"duplicate tweet_id values: {dup_count}")

    section("CONVERSATION RECONSTRUCTION CHECK")
    tweet_id_set = set(df["tweet_id"])
    sample = df.loc[df["in_response_to_tweet_id"].notna()].sample(
        n=min(2000, df["in_response_to_tweet_id"].notna().sum()), random_state=0
    )
    resolvable = sample["in_response_to_tweet_id"].isin(tweet_id_set).sum()
    print(
        f"sample of {len(sample):,} rows with in_response_to_tweet_id set: "
        f"{resolvable:,} ({resolvable / len(sample):.1%}) resolve to a tweet_id present in this file"
    )


if __name__ == "__main__":
    main()
