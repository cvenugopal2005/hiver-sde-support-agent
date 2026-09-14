"""Step 15 (addendum): human review tool for LLM-PROPOSED golden-set labels.

data/processed/golden_review.csv holds one row per previously-blank
golden_eval.csv example, with an LLM-generated PROPOSED label
(proposed_intent, proposed_response_behavior, proposed_handling,
labeling_reason) produced by reading the full conversation_context against
configs/golden_labeling_guide.md and configs/intent_taxonomy.md. These
proposals were NOT written into golden_eval.csv directly and are NOT ground
truth on their own -- this script is how a human turns a proposal into a
real, approved label.

For each row you can:
    y   approve all 3 proposed values as-is
    n   manually enter your own intent / response behavior / handling
    s   skip this row for now (come back later)
    q   save progress and quit

Whichever way a row is decided (y or n), the APPROVED values are written to
BOTH golden_review.csv (approved_intent/approved_response_behavior/
approved_handling) AND golden_eval.csv (intent/expected_response_behavior/
expected_handling), with a labeling_notes entry in golden_eval.csv that
records provenance (approved-as-proposed vs. manually overridden) so it's
always auditable which golden labels came from a reviewed LLM proposal and
which were manually corrected. Saved after every decided row -- safe to
stop and resume.

This script NEVER writes a label into golden_eval.csv without an explicit
y/n decision from whoever is running it interactively. The 9 rows that were
already hand-labeled before this review process exist only in
golden_eval.csv (not in golden_review.csv), so this script cannot touch
them -- there is nothing in golden_review.csv to match their tweet_ids.

Usage:
    python scripts/review_golden_labels.py            # start/resume review
    python scripts/review_golden_labels.py --report    # progress report only
"""

from __future__ import annotations

import sys
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
    "direct_instruction",
    "clarifying_question",
    "dm_handoff",
    "escalation",
    "confirmation/status",
    "unclear",
]

HANDLING_OPTIONS = ["AUTO-HANDLE", "ESCALATE"]


def load_review_df() -> pd.DataFrame:
    if not REVIEW_PATH.exists():
        raise SystemExit(f"{REVIEW_PATH} not found -- nothing to review.")
    return pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)


def load_golden_df() -> pd.DataFrame:
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"{GOLDEN_PATH} not found -- run scripts/prepare_golden_set.py first (Step 14).")
    return pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False)


def save(review_df: pd.DataFrame, golden_df: pd.DataFrame) -> None:
    review_df.to_csv(REVIEW_PATH, index=False)
    golden_df.to_csv(GOLDEN_PATH, index=False)


def is_reviewed(row: pd.Series) -> bool:
    return bool(row["approved_intent"].strip())


def print_report(review_df: pd.DataFrame) -> None:
    total = len(review_df)
    reviewed_mask = review_df.apply(is_reviewed, axis=1)
    approved = int(reviewed_mask.sum())
    remaining = total - approved

    print(f"\nTotal proposed rows: {total}")
    print(f"Approved (reviewed): {approved}")
    print(f"Remaining: {remaining}")

    print("\nIntent distribution (approved rows):")
    counts = review_df.loc[reviewed_mask, "approved_intent"].value_counts()
    for intent in INTENTS:
        print(f"  {intent}: {int(counts.get(intent, 0))}")
    print(f"  UNCLEAR: {int(counts.get('UNCLEAR', 0))}")

    print("\nExpected handling distribution (approved rows):")
    handling_counts = review_df.loc[reviewed_mask, "approved_handling"].value_counts()
    for option in HANDLING_OPTIONS:
        print(f"  {option}: {int(handling_counts.get(option, 0))}")

    if approved:
        n_overridden = int((review_df.loc[reviewed_mask, "approved_intent"] != review_df.loc[reviewed_mask, "proposed_intent"]).sum())
        print(f"\nOf the approved rows, {n_overridden} had the intent manually changed from the proposal.")


def print_menu() -> None:
    print("\nFor each row: y = approve proposal as-is, n = manually relabel,")
    print("s = skip for now, q = save and quit.")


def manual_label() -> tuple[str, str, str]:
    print("Intent:")
    for i, name in enumerate(INTENTS, start=1):
        print(f"  {i} {name}")
    print("  u UNCLEAR")
    while True:
        choice = input("Intent [1-8/u]: ").strip().lower()
        if choice == "u":
            intent = "UNCLEAR"
            break
        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            intent = INTENTS[int(choice) - 1]
            break
        print("Not a valid choice, try again.")

    print("Expected response behavior:")
    for i, name in enumerate(RESPONSE_BEHAVIORS, start=1):
        print(f"  {i} {name}")
    while True:
        choice = input("Expected response behavior [1-6]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(RESPONSE_BEHAVIORS):
            behavior = RESPONSE_BEHAVIORS[int(choice) - 1]
            break
        print("Not a valid choice, try again.")

    print("Expected handling:")
    for i, name in enumerate(HANDLING_OPTIONS, start=1):
        print(f"  {i} {name}")
    while True:
        choice = input("Expected handling [1-2]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(HANDLING_OPTIONS):
            handling = HANDLING_OPTIONS[int(choice) - 1]
            break
        print("Not a valid choice, try again.")

    return intent, behavior, handling


def apply_decision(
    review_df: pd.DataFrame,
    golden_df: pd.DataFrame,
    review_idx,
    tweet_id: str,
    intent: str,
    behavior: str,
    handling: str,
    note: str,
) -> None:
    review_df.loc[review_idx, "approved_intent"] = intent
    review_df.loc[review_idx, "approved_response_behavior"] = behavior
    review_df.loc[review_idx, "approved_handling"] = handling
    review_df.loc[review_idx, "labeling_notes"] = note

    golden_idx = golden_df.index[golden_df["tweet_id"] == tweet_id]
    if len(golden_idx) == 0:
        print(f"WARNING: tweet_id {tweet_id} not found in golden_eval.csv -- not written there.")
        return
    golden_df.loc[golden_idx, "intent"] = intent
    golden_df.loc[golden_idx, "expected_response_behavior"] = behavior
    golden_df.loc[golden_idx, "expected_handling"] = handling
    golden_df.loc[golden_idx, "labeling_notes"] = note


def review_loop(review_df: pd.DataFrame, golden_df: pd.DataFrame) -> None:
    remaining_idx = review_df.index[~review_df.apply(is_reviewed, axis=1)].tolist()
    if not remaining_idx:
        print("Nothing left to review. Run with --report to see the summary.")
        return

    print_menu()
    for position, idx in enumerate(remaining_idx, start=1):
        row = review_df.loc[idx]
        print(f"\n{'-' * 78}")
        print(f"[{position}/{len(remaining_idx)}] tweet_id={row['tweet_id']}")
        print(row["conversation_context"])
        print("-" * 78)
        print(f"PROPOSED intent:            {row['proposed_intent']}")
        print(f"PROPOSED response behavior: {row['proposed_response_behavior']}")
        print(f"PROPOSED handling:          {row['proposed_handling']}")
        print(f"REASON:                     {row['labeling_reason']}")

        try:
            choice = input("\n[y]approve / [n]relabel / [s]kip / [q]uit: ").strip().lower()
        except EOFError:
            choice = "q"

        if choice == "q":
            save(review_df, golden_df)
            left = len(remaining_idx) - position + 1
            print(f"Saved progress. {left} row(s) still unreviewed.")
            return

        if choice == "s":
            continue

        if choice == "y":
            note = f"Approved from LLM-proposed label. Reason: {row['labeling_reason']}"
            apply_decision(
                review_df, golden_df, idx, row["tweet_id"],
                row["proposed_intent"], row["proposed_response_behavior"], row["proposed_handling"],
                note,
            )
            save(review_df, golden_df)
            continue

        if choice == "n":
            intent, behavior, handling = manual_label()
            try:
                extra = input("Notes (optional, Enter to skip): ").strip()
            except EOFError:
                extra = ""
            note = "Manually relabeled by human reviewer (overrode LLM proposal)."
            if extra:
                note += f" Note: {extra}"
            apply_decision(review_df, golden_df, idx, row["tweet_id"], intent, behavior, handling, note)
            save(review_df, golden_df)
            continue

        print("Not a valid choice -- row skipped, still unreviewed.")

    save(review_df, golden_df)
    print("\nAll rows reviewed. Run with --report to see the summary.")


def main() -> None:
    review_df = load_review_df()
    if "--report" in sys.argv:
        print_report(review_df)
        return
    golden_df = load_golden_df()
    review_loop(review_df, golden_df)
    print_report(review_df)


if __name__ == "__main__":
    main()
