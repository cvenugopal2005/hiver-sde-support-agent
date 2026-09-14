"""Step 15: interactive CLI helper for hand-labeling
data/processed/golden_eval.csv -- the 200-example REQUIRED golden
evaluation set.

Run this yourself in a real terminal -- it needs live keyboard input.
Claude does not run this to assign labels; every label written to disk by
this tool comes from whoever typed it in interactively. No classifier, LLM,
or keyword rule ever touches these labels -- they are the human ground
truth Hiver evaluation is measured against, so getting them from anywhere
but a human would make that evaluation meaningless.

See configs/golden_labeling_guide.md for how to judge each field.

Progress is saved to disk after every fully-answered row, so it is always
safe to stop and resume later -- nothing is lost. A row is only marked
"labeled" once all three fields (intent, expected_response_behavior,
expected_handling) are filled in; partially-answered rows (e.g. you
answered intent then quit) are discarded, not half-saved, so the row stays
cleanly pending for next time.

Unlike scripts/label_training_data.py (Step 7), an invalid keystroke here
RE-ASKS the same question instead of silently skipping the row -- this file
is higher-stakes ground truth, so a stray keypress shouldn't cost a label.

Usage:
    python scripts/label_golden_set.py            # start/resume labeling
    python scripts/label_golden_set.py --report    # progress report only, no prompts

Per-row controls (shown on screen too):
    1-8 / u   answer the current question
    s         skip this row entirely, move to the next one
    q         save progress and quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "golden_eval.csv"

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

REQUIRED_COLUMNS = ["intent", "expected_response_behavior", "expected_handling", "labeling_notes"]


def load_working_df() -> pd.DataFrame:
    if not PATH.exists():
        raise SystemExit(f"{PATH} not found -- run scripts/prepare_golden_set.py first (Step 14).")
    df = pd.read_csv(PATH, dtype=str, keep_default_na=False)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def save(df: pd.DataFrame) -> None:
    df.to_csv(PATH, index=False)


def is_labeled(row: pd.Series) -> bool:
    return bool(row["intent"].strip()) and bool(row["expected_response_behavior"].strip()) and bool(row["expected_handling"].strip())


def print_report(df: pd.DataFrame) -> None:
    total = len(df)
    labeled_mask = df.apply(is_labeled, axis=1)
    labeled = int(labeled_mask.sum())
    remaining = total - labeled

    print(f"\nTotal rows: {total}")
    print(f"Labeled (all 3 fields filled): {labeled}")
    print(f"Remaining: {remaining}")

    print("\nIntent distribution (labeled rows):")
    intent_counts = df.loc[labeled_mask, "intent"].value_counts()
    for intent in INTENTS:
        print(f"  {intent}: {int(intent_counts.get(intent, 0))}")
    print(f"  UNCLEAR: {int(intent_counts.get('UNCLEAR', 0))}")
    known_intents = set(INTENTS) | {"UNCLEAR"}
    unexpected = intent_counts[~intent_counts.index.isin(known_intents)]
    if not unexpected.empty:
        print("  (unexpected values -- check for typos):", dict(unexpected))

    print("\nExpected response behavior distribution (labeled rows):")
    behavior_counts = df.loc[labeled_mask, "expected_response_behavior"].value_counts()
    for behavior in RESPONSE_BEHAVIORS:
        print(f"  {behavior}: {int(behavior_counts.get(behavior, 0))}")
    known_behaviors = set(RESPONSE_BEHAVIORS)
    unexpected_b = behavior_counts[~behavior_counts.index.isin(known_behaviors)]
    if not unexpected_b.empty:
        print("  (unexpected values -- check for typos):", dict(unexpected_b))

    print("\nExpected handling distribution (labeled rows):")
    handling_counts = df.loc[labeled_mask, "expected_handling"].value_counts()
    for option in HANDLING_OPTIONS:
        print(f"  {option}: {int(handling_counts.get(option, 0))}")
    known_handling = set(HANDLING_OPTIONS)
    unexpected_h = handling_counts[~handling_counts.index.isin(known_handling)]
    if not unexpected_h.empty:
        print("  (unexpected values -- check for typos):", dict(unexpected_h))


def print_menu() -> None:
    print("\nFor each row you'll answer 3 questions: intent, expected response")
    print("behavior, expected handling. At any prompt:")
    print("  s   skip this row entirely (comes back later)")
    print("  q   save progress and quit")


def ask(prompt: str, valid_keys: set[str]) -> str | None:
    """Loops until a valid key, 's', or 'q' is entered. Returns the raw key,
    or None if the user chose to quit (caller handles saving)."""
    while True:
        try:
            choice = input(prompt).strip()
        except EOFError:
            return "q"
        lower = choice.lower()
        if lower in ("s", "q"):
            return lower
        if lower in valid_keys:
            return lower
        print(f"Not a valid choice -- try again ({', '.join(sorted(valid_keys))}, s, or q).")


def label_loop(df: pd.DataFrame) -> None:
    remaining_idx = df.index[~df.apply(is_labeled, axis=1)].tolist()
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

        # --- 1. Intent ---
        print("Intent:")
        for i, name in enumerate(INTENTS, start=1):
            print(f"  {i} {name}")
        print("  u UNCLEAR")
        choice = ask("Intent [1-8/u/s/q]: ", {str(i) for i in range(1, 9)} | {"u"})
        if choice == "q":
            save(df)
            left = len(remaining_idx) - position + 1
            print(f"Saved progress to {PATH}. {left} row(s) still unlabeled.")
            return
        if choice == "s":
            continue
        intent = "UNCLEAR" if choice == "u" else INTENTS[int(choice) - 1]

        # --- 2. Expected response behavior ---
        print("Expected response behavior:")
        for i, name in enumerate(RESPONSE_BEHAVIORS, start=1):
            print(f"  {i} {name}")
        choice = ask("Expected response behavior [1-6/s/q]: ", {str(i) for i in range(1, 7)})
        if choice == "q":
            save(df)
            left = len(remaining_idx) - position + 1
            print(f"Saved progress to {PATH}. {left} row(s) still unlabeled.")
            return
        if choice == "s":
            continue
        expected_response_behavior = RESPONSE_BEHAVIORS[int(choice) - 1]

        # --- 3. Expected handling ---
        print("Expected handling:")
        for i, name in enumerate(HANDLING_OPTIONS, start=1):
            print(f"  {i} {name}")
        choice = ask("Expected handling [1-2/s/q]: ", {"1", "2"})
        if choice == "q":
            save(df)
            left = len(remaining_idx) - position + 1
            print(f"Saved progress to {PATH}. {left} row(s) still unlabeled.")
            return
        if choice == "s":
            continue
        expected_handling = HANDLING_OPTIONS[int(choice) - 1]

        try:
            note = input("Notes (optional, Enter to skip): ").strip()
        except EOFError:
            note = ""

        df.loc[idx, "intent"] = intent
        df.loc[idx, "expected_response_behavior"] = expected_response_behavior
        df.loc[idx, "expected_handling"] = expected_handling
        df.loc[idx, "labeling_notes"] = note
        save(df)  # progressive save after every fully-labeled row

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
