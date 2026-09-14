"""Evidence-based brand candidate analysis (Step 2B).

Loads data/raw/twcs.csv once, reconstructs conversation threads from the
`in_response_to_tweet_id` parent pointer (each row has at most one parent),
and reports per-brand volume, conversation-depth, resolution-pair, and
data-quality metrics for the 108 support accounts identified in Step 2A.

Thread reconstruction method (the "simplest correct approach"):
Every row points at its own parent via `in_response_to_tweet_id`. That is
sufficient to group rows into weakly-connected conversation threads by
walking each row up to its root (the ancestor with no resolvable parent).
`response_tweet_id` encodes the same edges in the opposite direction (and can
list several children per row, comma-separated) — it is redundant for
grouping into threads, so it is not parsed here. This avoids the exact
pitfall the task warns about (assuming one response_tweet_id per row).

Root-finding uses vectorized "pointer jumping": for a forest of parent
pointers, repeatedly following parent-of-parent doubles the jump distance
each round, so all nodes reach their root in O(log(max depth)) full-array
passes instead of a Python-level loop per row.

Does not select a final brand, define intents, or build any agent/model code.
Read-only against data/raw/twcs.csv.

Usage:
    python scripts/analyze_brands.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "twcs.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TOP_N_FOR_DEEP_DIVE = 7
MAX_JUMP_ROUNDS = 25  # 2^25 >> any realistic conversation depth; a safety cap, not a tuning knob
EXAMPLE_TEXT_TRUNCATE = 200


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])


def build_thread_roots(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (root_idx, dangling_parent) arrays of length n.

    root_idx[i] = row index of the thread root that row i belongs to.
    dangling_parent[i] = True if row i declares an in_response_to_tweet_id
    that does not resolve to any tweet_id present in this file.
    """
    n = len(df)
    tweet_id = pd.to_numeric(df["tweet_id"], errors="coerce")
    if tweet_id.isna().any():
        raise SystemExit("Non-numeric tweet_id values found; aborting.")
    tweet_id = tweet_id.astype("int64").to_numpy()

    parent_raw = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    had_parent_field = parent_raw.notna().to_numpy()
    parent_vals = parent_raw.fillna(-1).astype("int64").to_numpy()

    # Vectorized tweet_id -> row index lookup (tweet_id has gaps, so this
    # cannot be plain positional indexing; a hash-indexed Series does it in
    # one pass instead of a 2.8M-iteration Python dict loop).
    pos = pd.Series(np.arange(n, dtype="int64"), index=tweet_id)
    matched = pos.reindex(parent_vals).to_numpy()  # float64, NaN where parent_vals not found
    parent_unresolved = np.isnan(matched)
    parent_idx = np.where(parent_unresolved, np.arange(n), matched).astype("int64")

    dangling_parent = had_parent_field & parent_unresolved

    root = parent_idx.copy()
    for _ in range(MAX_JUMP_ROUNDS):
        new_root = root[root]
        if np.array_equal(new_root, root):
            break
        root = new_root

    return root, dangling_parent


def compute_thread_stats(df: pd.DataFrame, root: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            "root": root,
            "inbound": df["inbound"].to_numpy(),
            "author_id": df["author_id"].to_numpy(),
        }
    )
    size = tmp.groupby("root").size().rename("size")
    support = tmp[tmp["inbound"] == "False"]
    n_support = support.groupby("root").size()
    n_customer = tmp[tmp["inbound"] == "True"].groupby("root").size()
    brand_first = support.groupby("root")["author_id"].first()
    brand_nunique = support.groupby("root")["author_id"].nunique()

    stats = pd.DataFrame({"size": size})
    stats["n_support"] = n_support.reindex(stats.index).fillna(0).astype(int)
    stats["n_customer"] = n_customer.reindex(stats.index).fillna(0).astype(int)
    stats["brand"] = brand_first.reindex(stats.index)
    stats["n_distinct_brands"] = brand_nunique.reindex(stats.index).fillna(0).astype(int)
    return stats


def per_brand_metrics(df: pd.DataFrame, stats: pd.DataFrame, dangling: np.ndarray) -> pd.DataFrame:
    brands = sorted(df.loc[df["inbound"] == "False", "author_id"].unique())
    raw_support_counts = df.loc[df["inbound"] == "False", "author_id"].value_counts()

    rows = []
    for b in brands:
        tb = stats[stats["brand"] == b]
        brand_row_mask = (df["author_id"] == b).to_numpy()
        n_threads = len(tb)
        sizes = tb["size"]
        rows.append(
            {
                "brand": b,
                "n_threads": n_threads,
                "support_tweets": int(raw_support_counts.get(b, 0)),
                "support_tweets_in_threads": int(tb["n_support"].sum()),
                "customer_tweets_in_threads": int(tb["n_customer"].sum()),
                "total_tweets_in_threads": int(sizes.sum()),
                "resolution_pairs": int((tb["n_customer"] > 0).sum()),
                "median_len": float(sizes.median()) if n_threads else 0.0,
                "mean_len": float(sizes.mean()) if n_threads else 0.0,
                "pct_2plus": float((sizes >= 2).mean() * 100) if n_threads else 0.0,
                "pct_3plus": float((sizes >= 3).mean() * 100) if n_threads else 0.0,
                "pct_4plus": float((sizes >= 4).mean() * 100) if n_threads else 0.0,
                "max_len": int(sizes.max()) if n_threads else 0,
                "isolated_pct": float((sizes == 1).mean() * 100) if n_threads else 0.0,
                "multi_brand_thread_pct": float((tb["n_distinct_brands"] > 1).mean() * 100) if n_threads else 0.0,
                "dangling_pct": float(dangling[brand_row_mask].mean() * 100) if brand_row_mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_example_conversations(df: pd.DataFrame, root: np.ndarray, brand: str, n_examples: int = 2) -> None:
    tmp = pd.DataFrame(
        {
            "root": root,
            "inbound": df["inbound"].to_numpy(),
            "author_id": df["author_id"].to_numpy(),
        }
    )
    support = tmp[tmp["inbound"] == "False"]
    brand_roots = support.loc[support["author_id"] == brand, "root"]
    sizes = tmp.groupby("root").size()
    candidate_roots = [r for r in brand_roots.unique() if 3 <= sizes.get(r, 0) <= 5]
    chosen = candidate_roots[:n_examples] if candidate_roots else list(brand_roots.unique()[:n_examples])

    for k, r in enumerate(chosen, start=1):
        idx = np.where(root == r)[0]
        thread = df.iloc[idx].copy()
        thread["created_at_parsed"] = pd.to_datetime(thread["created_at"], errors="coerce")
        thread = thread.sort_values("created_at_parsed")
        print(f"--- {brand} example {k} (thread size={len(thread)}) ---")
        for _, row in thread.iterrows():
            role = "SUPPORT" if row["inbound"] == "False" else "customer"
            text = row["text"].replace("\n", " ")
            if len(text) > EXAMPLE_TEXT_TRUNCATE:
                text = text[:EXAMPLE_TEXT_TRUNCATE] + "…"
            print(f"  [{role:7s}] {row['author_id']}: {text}")
        print()


def handle_signal_check(df: pd.DataFrame, root: np.ndarray, brand: str) -> tuple[int, float]:
    tmp = pd.DataFrame(
        {
            "root": root,
            "inbound": df["inbound"].to_numpy(),
            "author_id": df["author_id"].to_numpy(),
        }
    )
    support = tmp[tmp["inbound"] == "False"]
    brand_roots = set(support.loc[support["author_id"] == brand, "root"].unique())
    cust_mask = (tmp["inbound"] == "True") & tmp["root"].isin(brand_roots)
    cust_idx = np.where(cust_mask.to_numpy())[0]
    if len(cust_idx) == 0:
        return 0, 0.0
    texts = df["text"].to_numpy()[cust_idx]
    handle = f"@{brand}".lower()
    hits = sum(1 for t in texts if t.strip().lower().startswith(handle))
    return len(cust_idx), (hits / len(cust_idx)) * 100


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"Dataset not found at {RAW_PATH}")

    section("LOADING DATA")
    df = load_data()
    print(f"rows: {len(df):,}")

    section("TASK 1: SUPPORT/BRAND ACCOUNTS (derived from data, not hard-coded)")
    brands = sorted(df.loc[df["inbound"] == "False", "author_id"].unique())
    print(f"distinct support accounts (inbound=False author_id): {len(brands)}")

    section("BUILDING CONVERSATION THREADS (parent-pointer reconstruction)")
    root, dangling = build_thread_roots(df)
    n_threads_total = len(np.unique(root))
    print(f"total distinct threads in dataset: {n_threads_total:,}")
    print(f"rows with a dangling in_response_to_tweet_id (declared parent not found): {dangling.sum():,}")

    stats = compute_thread_stats(df, root)

    section("TASKS 2-5: PER-BRAND METRICS (raw, no scoring formula)")
    metrics = per_brand_metrics(df, stats, dangling)
    metrics_sorted = metrics.sort_values("resolution_pairs", ascending=False).reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(
            metrics_sorted[
                [
                    "brand",
                    "n_threads",
                    "support_tweets",
                    "customer_tweets_in_threads",
                    "resolution_pairs",
                    "median_len",
                    "mean_len",
                    "pct_2plus",
                    "pct_3plus",
                    "pct_4plus",
                    "max_len",
                    "isolated_pct",
                    "dangling_pct",
                    "multi_brand_thread_pct",
                ]
            ].round(1).to_string(index=False)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "brand_metrics.csv"
    metrics_sorted.to_csv(out_csv, index=False)
    print(f"\nfull per-brand metrics written to: {out_csv} (gitignored, not for commit)")

    section(f"TASK 6: TOP {TOP_N_FOR_DEEP_DIVE} CANDIDATES BY resolution_pairs (raw ranking signal, not final)")
    top = metrics_sorted.head(TOP_N_FOR_DEEP_DIVE)
    print(top[["brand", "n_threads", "resolution_pairs", "pct_3plus", "dangling_pct"]].round(1).to_string(index=False))

    section("TASK 7: REPRESENTATIVE CONVERSATION EXAMPLES (real rows, truncated text)")
    for b in top["brand"]:
        print_example_conversations(df, root, b, n_examples=1)

    section("TASK 8: @HANDLE SIGNAL VALIDATION (secondary check, not primary method)")
    for b in top["brand"]:
        n_cust, pct = handle_signal_check(df, root, b)
        print(f"{b}: customer messages in its threads = {n_cust:,}, starting with '@{b}' (case-insensitive) = {pct:.1f}%")


if __name__ == "__main__":
    main()
