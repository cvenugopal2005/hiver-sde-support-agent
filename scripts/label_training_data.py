"""Step 7: interactive CLI helper for manually labeling
data/processed/intent_training_sample.csv.

Run this yourself in a real terminal -- it needs live keyboard input.
Claude does not run this to assign labels; every label written to disk by
this tool comes from whoever typed it in interactively.

Progress is saved to disk after every single label (and on skip/quit), so
it is always safe to stop and resume later -- nothing is lost.

Usage:
    python scripts/label_training_data.py            # start/resume labeling
    python scripts/label_training_data.py --report    # progress report only, no prompts

Labeling controls (shown on screen too):
    1-8      assign one of the 8 taxonomy intents
    u        UNCLEAR
    <Enter>  skip this row for now (come back later)
    q        save progress and quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "intent_training_sample.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "intent_training_sample_labeled.csv"

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


def load_working_df() -> pd.DataFrame:
    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH, dtype=str, keep_default_na=False)
        print(f"Resuming from {OUT_PATH}")
    else:
        df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)
        df["intent"] = ""
        df["labeling_notes"] = ""
        print(f"Starting fresh from {IN_PATH}")
    return df


def save(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


def print_report(df: pd.DataFrame) -> None:
    total = len(df)
    labeled_mask = df["intent"].str.strip() != ""
    labeled = int(labeled_mask.sum())
    remaining = total - labeled

    print(f"\nTotal rows: {total}")
    print(f"Labeled: {labeled}")
    print(f"Remaining: {remaining}")

    counts = df.loc[labeled_mask, "intent"].value_counts()
    print("\nCount per intent:")
    for intent in INTENTS:
        print(f"  {intent}: {int(counts.get(intent, 0))}")
    print(f"  UNCLEAR: {int(counts.get('UNCLEAR', 0))}")

    known = set(INTENTS) | {"UNCLEAR"}
    unexpected = counts[~counts.index.isin(known)]
    if not unexpected.empty:
        print("  (unexpected values found -- check for typos):")
        for k, v in unexpected.items():
            print(f"    {k!r}: {v}")


def print_menu() -> None:
    print("\nIntents:")
    for i, name in enumerate(INTENTS, start=1):
        print(f"  {i}. {name}")
    print("  u        UNCLEAR")
    print("  <Enter>  skip for now")
    print("  q        save and quit")


def label_loop(df: pd.DataFrame) -> None:
    remaining_idx = df.index[df["intent"].str.strip() == ""].tolist()
    if not remaining_idx:
        print("Nothing left to label. Run with --report to see the summary.")
        return

    print_menu()
    for position, idx in enumerate(remaining_idx, start=1):
        row = df.loc[idx]
        print(f"\n{'-' * 78}")
        print(f"[{position}/{len(remaining_idx)}] tweet_id={row['tweet_id']}")
        print(row["conversation_context"])
        print("-" * 78)

        try:
            choice = input("Intent [1-8/u/q/Enter=skip]: ").strip()
        except EOFError:
            choice = "q"

        if choice.lower() == "q":
            save(df)
            left = len(remaining_idx) - position + 1
            print(f"Saved progress to {OUT_PATH}. {left} row(s) still unlabeled.")
            return
        if choice == "":
            continue
        if choice.lower() == "u":
            label = "UNCLEAR"
        elif choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            label = INTENTS[int(choice) - 1]
        else:
            print("Not a valid choice -- row skipped, still unlabeled.")
            continue

        try:
            note = input("Notes (optional, Enter to skip): ").strip()
        except EOFError:
            note = ""

        df.loc[idx, "intent"] = label
        df.loc[idx, "labeling_notes"] = note
        save(df)  # progressive save after every single label

    save(df)
    print("\nAll rows labeled. Run with --report to see the summary.")


def main() -> None:
    df = load_working_df()
    if "--report" in sys.argv:
        print_report(df)
        return
    label_loop(df)
    print_report(df)


if __name__ == "__main__":
    main()
