"""Step 14: build the REQUIRED golden evaluation set (200 examples).

This is a completely separate draw from data/raw/twcs.csv than every prior
sample in this project. It reuses thread reconstruction and the AppleSupport
brand filter (scripts/analyze_brands.py, scripts/explore_applesupport.py)
but with a brand-new seed, AND it explicitly excludes every thread already
used to build the 1000-thread Step 5 labeling pool (of which the 450-example
silver TRAINING set is a subset -- verified: every training tweet_id is in
that 1000-thread pool). So this is independent of the training data by
construction, not just by a different random seed.

NO LABELS ARE ASSIGNED HERE. intent, expected_response_behavior,
expected_handling, and labeling_notes are all written as empty strings.
No LLM, no keyword rule, no trained classifier, and no reuse of the Step 8/9
silver labels is used anywhere in this script -- labeling is done afterwards
by a human, using configs/golden_labeling_guide.md. This set exists
specifically so the classifier, retriever, and decision logic built in
Steps 10-13 can later be measured against ground truth that never
influenced any of them.

Output (NOT gitignored on purpose, same reasoning as intent_labeling_sample.csv
in Step 5 -- see the printed note at the end):
    data/processed/golden_eval.csv

Usage:
    python scripts/prepare_golden_set.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_brands import build_thread_roots, compute_thread_stats, load_data  # noqa: E402
from explore_applesupport import BRAND, build_sample  # noqa: E402

SEED = 4242  # distinct from every prior seed used in this project (42, 123, 2024, 77)
N_THREADS = 200

# The exact seed/size used to build the Step 5 labeling pool that the 450
# training examples were drawn from (scripts/prepare_intent_data.py). Reused
# here ONLY to recompute which thread roots to exclude -- never to sample.
TRAINING_POOL_SEED = 2024
TRAINING_POOL_N_THREADS = 1000

MAX_MESSAGES_IN_CONTEXT = 10
MESSAGE_TRUNCATE = 300

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "golden_eval.csv"
TRAINING_SAMPLE_PATH = DATA_DIR / "intent_training_sample.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def get_excluded_root_ids(df: pd.DataFrame, root: np.ndarray, stats: pd.DataFrame) -> set[int]:
    """Recomputes the exact 1000-thread pool Step 5 sampled from (same seed,
    same n_threads, same deterministic data load), so the golden set can
    exclude every one of those threads by construction."""
    training_pool = build_sample(df, root, stats, seed=TRAINING_POOL_SEED, n_threads=TRAINING_POOL_N_THREADS)
    return set(training_pool["_root"].unique().tolist())


def build_golden_sample(
    df: pd.DataFrame,
    root: np.ndarray,
    stats: pd.DataFrame,
    excluded_roots: set[int],
) -> pd.DataFrame:
    brand_threads = stats[
        (stats["brand"] == BRAND) & (stats["n_customer"] > 0) & (stats["n_support"] > 0)
    ]
    candidate_root_ids = brand_threads.index.to_numpy()
    excluded_arr = np.fromiter(excluded_roots, dtype=candidate_root_ids.dtype, count=len(excluded_roots))
    candidate_root_ids = candidate_root_ids[~np.isin(candidate_root_ids, excluded_arr)]

    rng = np.random.default_rng(SEED)
    n = min(N_THREADS, len(candidate_root_ids))
    chosen = rng.choice(candidate_root_ids, size=n, replace=False)

    mask = np.isin(root, chosen)
    sample_df = df.loc[mask].copy()
    sample_df["_root"] = root[mask]
    return sample_df


def build_golden_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_df = sample_df.copy()
    sample_df["_ts"] = pd.to_datetime(sample_df["created_at"], errors="coerce", format="mixed")

    rows = []
    for _root, thread in sample_df.groupby("_root"):
        thread = thread.sort_values("_ts")
        customer_rows = thread[thread["inbound"] == "True"]
        if len(customer_rows) == 0:
            continue  # should not happen given the qualifying-thread filter, but be defensive

        primary = customer_rows.iloc[0]

        context_lines = []
        for _, row in thread.head(MAX_MESSAGES_IN_CONTEXT).iterrows():
            label = "APPLESUPPORT" if row["inbound"] == "False" else "CUSTOMER"
            text = row["text"].replace("\n", " ")
            if len(text) > MESSAGE_TRUNCATE:
                text = text[:MESSAGE_TRUNCATE] + "…"
            context_lines.append(f"{label}: {text}")
        truncated_note = " [thread truncated]" if len(thread) > MAX_MESSAGES_IN_CONTEXT else ""
        conversation_context = "\n".join(context_lines) + truncated_note

        rows.append(
            {
                "tweet_id": primary["tweet_id"],
                "conversation_context": conversation_context,
                "intent": "",  # blank -- human labels this
                "expected_response_behavior": "",  # blank -- human labels this
                "expected_handling": "",  # blank -- human labels this
                "labeling_notes": "",  # blank -- human fills this in
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    section("LOADING DATA + RECONSTRUCTING THREADS (reused from analyze_brands.py)")
    df = load_data()
    root, _dangling = build_thread_roots(df)
    stats = compute_thread_stats(df, root)

    section(f"RECOMPUTING THE {TRAINING_POOL_N_THREADS}-THREAD TRAINING POOL TO EXCLUDE (seed={TRAINING_POOL_SEED})")
    excluded_roots = get_excluded_root_ids(df, root, stats)
    print(f"thread roots excluded (Step 5 pool, superset of the 450 silver-training examples): {len(excluded_roots)}")

    section(f"SAMPLING {N_THREADS} NEW APPLESUPPORT THREADS (seed={SEED}, excludes the training pool)")
    sample_df = build_golden_sample(df, root, stats, excluded_roots)
    golden_df = build_golden_rows(sample_df)

    section("INDEPENDENCE CHECKS")
    overlap_roots = set(sample_df["_root"].unique()) & excluded_roots
    print(f"golden thread roots overlapping the excluded training pool: {len(overlap_roots)} (must be 0)")
    assert len(overlap_roots) == 0, "golden set overlaps the training pool -- aborting"

    if TRAINING_SAMPLE_PATH.exists():
        training_tweet_ids = set(pd.read_csv(TRAINING_SAMPLE_PATH, dtype=str)["tweet_id"])
        golden_tweet_ids = set(golden_df["tweet_id"].astype(str))
        overlap_tweets = training_tweet_ids & golden_tweet_ids
        print(f"golden tweet_ids overlapping the 450 training tweet_ids: {len(overlap_tweets)} (must be 0)")
        assert len(overlap_tweets) == 0, "golden set overlaps the 450 training examples -- aborting"

    for col in ("intent", "expected_response_behavior", "expected_handling", "labeling_notes"):
        n_nonblank = (golden_df[col] != "").sum()
        assert n_nonblank == 0, f"column {col!r} has {n_nonblank} non-blank values -- labels must not be auto-assigned"
    print("confirmed: intent / expected_response_behavior / expected_handling / labeling_notes are all blank "
          "(no automatic labeling of any kind was performed)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    golden_df.to_csv(OUT_PATH, index=False)

    section("RESULT")
    print(f"brand: {BRAND}")
    print(f"rows written: {len(golden_df)}")
    print(f"output: {OUT_PATH}")
    print(
        "\nNOTE on .gitignore: data/processed/* is currently blanket-gitignored "
        "(same situation as intent_labeling_sample.csv in Step 5). This file "
        "becomes the real golden ground-truth eval set once hand-labeled -- it "
        "will need a .gitignore exception (or a move out of data/processed/) "
        "before it can ever be committed. Not changed here since this step "
        "makes no Git commits or repo-structure changes."
    )


if __name__ == "__main__":
    main()
