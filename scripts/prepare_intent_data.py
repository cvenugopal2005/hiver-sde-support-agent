"""Step 5: build a blank-intent, manually-labelable dataset for the
8-intent AppleSupport classifier.

Reuses (does not reimplement) thread reconstruction from
scripts/analyze_brands.py and the AppleSupport thread-sampling method from
scripts/explore_applesupport.py, with a new seed so this is an independent
draw from the Step 3A/3B exploration samples.

This script does NOT assign any intent labels. The `intent` and
`labeling_notes` columns are written empty on purpose -- labeling is done by
a human afterwards, using configs/labeling_guide.md. No LLM, no keyword
matching, no model of any kind is used to produce labels here.

One row = one AppleSupport conversation thread (not one raw tweet), so the
labeler has full conversational context. `customer_text` is the FIRST
customer message in the thread (the one that states the issue);
`conversation_context` is the full thread transcript in chronological order.

Output (NOT gitignored on purpose -- unlike the Step 3A/3B exploration
samples, this file is meant to become the real hand-labeled training set,
so it should eventually be committed once labeled; see the .gitignore note
printed at the end of this script):
    data/processed/intent_labeling_sample.csv

Usage:
    python scripts/prepare_intent_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_brands import build_thread_roots, compute_thread_stats, load_data  # noqa: E402
from explore_applesupport import BRAND, build_sample  # noqa: E402

SEED = 2024  # distinct from Step 3A (seed=42) and Step 3B (seed=123)
N_THREADS = 1000
MAX_MESSAGES_IN_CONTEXT = 10
MESSAGE_TRUNCATE = 300

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "intent_labeling_sample.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_labeling_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample_df = sample_df.copy()
    # Parsed once, vectorized, over the whole sample instead of once per
    # thread group (faster, and avoids one dateutil-fallback warning per group).
    sample_df["_ts"] = pd.to_datetime(sample_df["created_at"], errors="coerce", format="mixed")

    rows = []
    for root, thread in sample_df.groupby("_root"):
        thread = thread.sort_values("_ts")

        customer_rows = thread[thread["inbound"] == "True"]
        support_rows = thread[thread["inbound"] == "False"]
        if len(customer_rows) == 0:
            continue  # should not happen given build_sample's filter, but be defensive

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
                "thread_root_tweet_id": thread.iloc[0]["tweet_id"],
                "thread_size": len(thread),
                "num_customer_messages": len(customer_rows),
                "num_applesupport_messages": len(support_rows),
                "created_at": primary["created_at"],
                "customer_text": primary["text"].replace("\n", " "),
                "conversation_context": conversation_context,
                "intent": "",  # left blank for manual labeling
                "labeling_notes": "",  # left blank for manual labeling
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    section("LOADING DATA + RECONSTRUCTING THREADS (reused from analyze_brands.py)")
    df = load_data()
    root, _dangling = build_thread_roots(df)
    stats = compute_thread_stats(df, root)

    section(f"SAMPLING {N_THREADS} APPLESUPPORT THREADS (seed={SEED}, reused from explore_applesupport.py)")
    sample_df = build_sample(df, root, stats, seed=SEED, n_threads=N_THREADS)

    labeling_df = build_labeling_rows(sample_df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeling_df.to_csv(OUT_PATH, index=False)

    section("RESULT")
    print(f"brand: {BRAND}")
    print(f"rows written: {len(labeling_df)}")
    print(f"output: {OUT_PATH}")
    print(f"thread size distribution (rows to label):\n{labeling_df['thread_size'].value_counts().sort_index().to_string()}")
    print(
        "\nNOTE on .gitignore: data/processed/* is currently blanket-gitignored "
        "(see Step 1). This file is a work-in-progress labeling sheet right now, "
        "but once hand-labeled it becomes real ground-truth training data, not a "
        "throwaway artifact -- it will need a .gitignore exception (or a move out "
        "of data/processed/) before it can ever be committed. Not changed here "
        "since this step makes no Git commits or repo-structure changes."
    )


if __name__ == "__main__":
    main()
