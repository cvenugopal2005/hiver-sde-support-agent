"""Step 15 addendum #2: BULK auto-approve the 191 LLM-proposed golden-set
labels, without individual human review.

*** THIS IS A DELIBERATE DEVIATION FROM STEP 14/15'S REQUIREMENT THAT THE ***
*** GOLDEN EVALUATION SET BE HUMAN-LABELED / HUMAN-APPROVED. ***

Explicitly requested by the project owner to finish the golden set quickly,
with the explicit, upfront acknowledgment that the result is NOT a fully
hand-labeled evaluation set and must never be represented as satisfying
Hiver's hand-labeled requirement. See the STEP 15 ADDENDUM #2 note in
docs/project_notes.txt, which must accompany this file's output wherever
the golden set's provenance is described.

What this script does (nothing more):
    - Copies, for all 191 rows in data/processed/golden_review.csv:
          proposed_intent             -> intent
          proposed_response_behavior  -> expected_response_behavior
          proposed_handling           -> expected_handling
          labeling_reason             -> labeling_notes
      into the matching row (by tweet_id) of data/processed/golden_eval.csv.
    - Mirrors the same values into golden_review.csv's approved_* columns
      (marking them auto-approved, not individually reviewed) so that file's
      own state stays consistent with what actually happened.
    - Never touches the 9 rows that were already genuinely hand-labeled
      before this script ran (they have no matching row in
      golden_review.csv, so there is nothing for this script to overwrite
      there -- verified explicitly below, not just assumed).

Usage:
    python scripts/bulk_approve_golden_labels.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REVIEW_PATH = DATA_DIR / "golden_review.csv"
GOLDEN_PATH = DATA_DIR / "golden_eval.csv"

INTENTS = [
    "Battery & Performance",
    "Connectivity",
    "Apple Music / iCloud Sync & Data Loss",
    "Account, Purchases & Billing",
    "Hardware, Repair & Warranty",
    "Messaging & Communication",
    "Software / App Bug Report",
    "General / Pre-Purchase Inquiry",
]
RESPONSE_BEHAVIORS = [
    "direct_instruction", "clarifying_question", "dm_handoff",
    "escalation", "confirmation/status", "unclear",
]
HANDLING_OPTIONS = ["AUTO-HANDLE", "ESCALATE"]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    review_df = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)
    golden_df = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False)

    section("PRE-CHECK")
    already_labeled_ids = set(golden_df.loc[golden_df["intent"] != "", "tweet_id"])
    already_approved_ids = set(review_df.loc[review_df["approved_intent"] != "", "tweet_id"])
    print(f"golden_eval.csv rows already human-labeled (must be left untouched): {len(already_labeled_ids)}")
    print(f"golden_review.csv rows already individually approved (e.g. via the interactive "
          f"review tool since the proposals were generated -- must be left untouched): {len(already_approved_ids)}")
    already_decided_ids = already_labeled_ids | already_approved_ids
    if already_decided_ids:
        print(f"tweet_id(s) already decided, skipping in this bulk run: {sorted(already_decided_ids)}")

    to_process = review_df[~review_df["tweet_id"].isin(already_decided_ids)]
    print(f"golden_review.csv proposal rows to bulk-approve now: {len(to_process)}")

    section("BULK-COPYING proposed_* -> golden_eval.csv (by tweet_id), skipping already-decided rows")
    n_written = 0
    for _, row in to_process.iterrows():
        golden_idx = golden_df.index[golden_df["tweet_id"] == row["tweet_id"]]
        assert len(golden_idx) == 1, f"tweet_id {row['tweet_id']} not found exactly once in golden_eval.csv"
        assert golden_df.loc[golden_idx, "intent"].iloc[0] == "", f"tweet_id {row['tweet_id']} unexpectedly already labeled -- aborting"
        golden_df.loc[golden_idx, "intent"] = row["proposed_intent"]
        golden_df.loc[golden_idx, "expected_response_behavior"] = row["proposed_response_behavior"]
        golden_df.loc[golden_idx, "expected_handling"] = row["proposed_handling"]
        golden_df.loc[golden_idx, "labeling_notes"] = row["labeling_reason"]

        review_idx = review_df.index[review_df["tweet_id"] == row["tweet_id"]]
        review_df.loc[review_idx, "approved_intent"] = row["proposed_intent"]
        review_df.loc[review_idx, "approved_response_behavior"] = row["proposed_response_behavior"]
        review_df.loc[review_idx, "approved_handling"] = row["proposed_handling"]
        review_df.loc[review_idx, "labeling_notes"] = (
            "Auto-approved in bulk WITHOUT individual human review "
            "(see docs/project_notes.txt, STEP 15 ADDENDUM #2). Reason given: " + row["labeling_reason"]
        )
        n_written += 1
    print(f"rows written into golden_eval.csv: {n_written}")

    golden_df.to_csv(GOLDEN_PATH, index=False)
    review_df.to_csv(REVIEW_PATH, index=False)

    section("VERIFICATION")
    reloaded = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False)
    assert len(reloaded) == 200, f"expected 200 rows, got {len(reloaded)}"
    print(f"golden_eval.csv row count: {len(reloaded)} (must be 200)")

    for col in ("intent", "expected_response_behavior", "expected_handling"):
        n_blank = (reloaded[col] == "").sum()
        print(f"blank '{col}' values: {n_blank} (must be 0)")
        assert n_blank == 0, f"{n_blank} rows still have a blank {col}"

    n_dupes = reloaded["tweet_id"].duplicated().sum()
    print(f"duplicate tweet_ids: {n_dupes} (must be 0)")
    assert n_dupes == 0, "duplicate tweet_ids found in golden_eval.csv"

    invalid_intent = reloaded[~reloaded["intent"].isin(INTENTS + ["UNCLEAR"])]
    invalid_behavior = reloaded[~reloaded["expected_response_behavior"].isin(RESPONSE_BEHAVIORS)]
    invalid_handling = reloaded[~reloaded["expected_handling"].isin(HANDLING_OPTIONS)]
    print(f"rows with an invalid intent value: {len(invalid_intent)} (must be 0)")
    print(f"rows with an invalid response-behavior value: {len(invalid_behavior)} (must be 0)")
    print(f"rows with an invalid handling value: {len(invalid_handling)} (must be 0)")
    assert len(invalid_intent) == 0 and len(invalid_behavior) == 0 and len(invalid_handling) == 0

    section("FINAL REPORT")
    # Computed from final cumulative file state (not "rows written this run"),
    # so this is correct whether the script has been run once or many times:
    # a row is auto-approved-without-review iff golden_review.csv marks it
    # that way; every other row (including ones never in golden_review.csv
    # at all) is genuinely human-decided.
    reloaded_review = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)
    auto_ids = set(
        reloaded_review.loc[
            reloaded_review["labeling_notes"].str.startswith("Auto-approved in bulk"), "tweet_id"
        ]
    )
    n_auto = len(auto_ids)
    n_human = len(reloaded) - n_auto
    print(f"Original human-labeled rows:        {n_human}")
    print(f"Automatically approved (LLM) rows:  {n_auto}")
    print(f"Total:                              {len(reloaded)}")
    print(f"(rows newly bulk-approved in this run: {n_written})")

    print("\nIntent distribution (all 200 rows):")
    counts = reloaded["intent"].value_counts()
    for intent in INTENTS + ["UNCLEAR"]:
        print(f"  {intent}: {int(counts.get(intent, 0))}")

    print("\nExpected response behavior distribution (all 200 rows):")
    counts = reloaded["expected_response_behavior"].value_counts()
    for behavior in RESPONSE_BEHAVIORS:
        print(f"  {behavior}: {int(counts.get(behavior, 0))}")

    print("\nExpected handling distribution (all 200 rows):")
    counts = reloaded["expected_handling"].value_counts()
    for option in HANDLING_OPTIONS:
        print(f"  {option}: {int(counts.get(option, 0))}")

    section("REMINDER (see docs/project_notes.txt)")
    print(
        f"Golden set contains {n_human} human-labelled examples and {n_auto} "
        "LLM-generated labels. It is therefore NOT a fully hand-labelled "
        "evaluation set and must not be represented as such."
    )


if __name__ == "__main__":
    main()
