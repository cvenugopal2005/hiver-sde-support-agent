"""Step 9: LLM-assisted refinement of the 450 rule-based ("silver") training
labels from Step 8.

IMPORTANT -- PROVENANCE OF THE LLM CLASSIFICATIONS:
This environment has no configured LLM API key (no ANTHROPIC_API_KEY /
OPENAI_API_KEY, no .env, no `claude` CLI on PATH). Rather than fabricate
results or block indefinitely, the classifications in
data/processed/llm_classification_raw.json were produced directly by
Claude (the assistant) acting AS the LLM classifier: each of the 450
conversations was read in full and classified according to the exact
prompt/rules encoded in this script's CLASSIFICATION_PROMPT_TEMPLATE, using
Claude subagents (independent LLM calls) dispatched in 6 batches of 75.
This script's job is the deterministic, reproducible part: loading the
source data, merging the (separately produced) classification results, and
generating the before/after comparison report. It does NOT call an API
itself. If a real API key becomes available later, `call_llm_api()` below
shows where a genuine HTTP call would replace the precomputed-results path
-- the prompt template is identical either way, so results should be
comparable.

These are LLM-ASSISTED SILVER LABELS for training only. They are NOT the
150-250-example hand-labeled GOLD evaluation set the assignment requires;
that must still be built and labeled independently, by a human.

Outputs:
    data/processed/intent_training_sample_silver.csv
        columns: tweet_id, conversation_context, intent, confidence, labeling_reason

Usage:
    python scripts/refine_training_labels.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SOURCE_CSV = DATA_DIR / "intent_training_sample.csv"
OLD_LABELED_CSV = DATA_DIR / "intent_training_sample_labeled.csv"  # Step 8 rule-based labels
LLM_RESULTS_JSON = DATA_DIR / "llm_classification_raw.json"
OUT_CSV = DATA_DIR / "intent_training_sample_silver.csv"

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
VALID_LABELS = set(INTENTS) | {"UNCLEAR"}

# The exact rules given to the LLM (subagents) for every batch of 75 rows.
# Kept here verbatim so the classification method is auditable and, with a
# real API key, exactly reproducible via call_llm_api().
CLASSIFICATION_PROMPT_TEMPLATE = """
You are classifying real AppleSupport customer-service Twitter conversations
into exactly one of 8 intents, or UNCLEAR ... (see configs/intent_taxonomy.md
and configs/labeling_guide.md for the full rule set: the 8 intent
definitions, the primary-intent priority order, the "I" autocorrect bug
rule, and the Pre-Purchase / UNCLEAR criteria). Read the FULL
conversation_context, not just the first line, before answering.
Return one JSON object per row: {"tweet_id", "intent", "confidence", "reason"}.
"""


def call_llm_api(conversation_context: str) -> dict:
    """Placeholder for a real LLM API call (e.g. Anthropic Messages API).

    Not used in this run -- no API key is configured in this environment.
    Left here so a future run with credentials can swap the data source
    (see main()) for a real per-row API call using CLASSIFICATION_PROMPT_TEMPLATE
    instead of the precomputed data/processed/llm_classification_raw.json.
    """
    raise NotImplementedError(
        "No LLM API key configured in this environment. Classifications for "
        "this run were produced via Claude subagents instead -- see "
        "data/processed/llm_classification_raw.json and the module docstring."
    )


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_llm_results() -> pd.DataFrame:
    with open(LLM_RESULTS_JSON) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    bad = df[~df["intent"].isin(VALID_LABELS)]
    if len(bad):
        raise SystemExit(f"Invalid intent labels found (not in the 8-intent taxonomy or UNCLEAR): {bad}")
    return df


def build_silver_csv() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_CSV, dtype=str)
    llm = load_llm_results()
    llm["tweet_id"] = llm["tweet_id"].astype(str)

    merged = source[["tweet_id", "conversation_context"]].merge(llm, on="tweet_id", how="left")
    if merged["intent"].isna().any():
        missing = merged[merged["intent"].isna()]["tweet_id"].tolist()
        raise SystemExit(f"Missing LLM classification for tweet_id(s): {missing}")

    out = merged.rename(columns={"reason": "labeling_reason"})[
        ["tweet_id", "conversation_context", "intent", "confidence", "labeling_reason"]
    ]
    out.to_csv(OUT_CSV, index=False)
    return out


def compare_reports(old_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    total = len(new_df)

    section("D. CLASS DISTRIBUTION -- OLD (rule-based, Step 8) vs NEW (LLM-assisted, Step 9)")
    old_counts = old_df["intent"].value_counts()
    new_counts = new_df["intent"].value_counts()
    print(f"{'intent':40s} {'old n (%)':>16s} {'new n (%)':>16s}")
    for intent in INTENTS + ["UNCLEAR"]:
        o = int(old_counts.get(intent, 0))
        n = int(new_counts.get(intent, 0))
        print(f"{intent:40s} {o:>4d} ({o/total*100:5.1f}%){'':>2s} {n:>4d} ({n/total*100:5.1f}%)")

    section("Changed labels (old rule-based label != new LLM-assisted label)")
    merged = old_df[["tweet_id", "conversation_context", "intent"]].rename(columns={"intent": "old_intent"}).merge(
        new_df[["tweet_id", "intent", "confidence", "labeling_reason"]].rename(columns={"intent": "new_intent"}),
        on="tweet_id",
    )
    changed = merged[merged["old_intent"] != merged["new_intent"]]
    print(f"changed: {len(changed)} / {total} ({len(changed) / total * 100:.1f}%)")

    section("E. 10 REPRESENTATIVE EXAMPLES WHERE OLD AND NEW LABELS DIFFER")
    for _, row in changed.head(10).iterrows():
        first_line = row["conversation_context"].splitlines()[0][:110]
        print(f"\ntweet_id={row['tweet_id']}")
        print(f"  text: {first_line}")
        print(f"  OLD (rule-based): {row['old_intent']}")
        print(f"  NEW (LLM-assisted, {row['confidence']} confidence): {row['new_intent']}")
        print(f"  reason: {row['labeling_reason']}")

    section("Examples still classified UNCLEAR (new labels)")
    still_unclear = new_df[new_df["intent"] == "UNCLEAR"]
    print(f"count: {len(still_unclear)} / {total} ({len(still_unclear) / total * 100:.1f}%)")
    for _, row in still_unclear.head(5).iterrows():
        first_line = row["conversation_context"].splitlines()[0][:110]
        print(f"  tweet_id={row['tweet_id']}: {first_line} -- {row['labeling_reason']}")

    section("Examples of ambiguous multi-issue cases (medium/low confidence)")
    ambiguous = new_df[new_df["confidence"].isin(["medium", "low"])]
    print(f"medium/low confidence count: {len(ambiguous)} / {total} ({len(ambiguous) / total * 100:.1f}%)")
    for _, row in ambiguous.head(5).iterrows():
        first_line = row["conversation_context"].splitlines()[0][:110]
        print(f"  tweet_id={row['tweet_id']} [{row['confidence']}] -> {row['intent']}: {first_line} -- {row['labeling_reason']}")


def main() -> None:
    section("LOADING SOURCE + OLD (RULE-BASED) LABELS + LLM CLASSIFICATION RESULTS")
    old_df = pd.read_csv(OLD_LABELED_CSV, dtype=str, keep_default_na=False)
    print(f"old (Step 8 rule-based) labeled rows: {len(old_df)}")

    new_df = build_silver_csv()
    print(f"new (Step 9 LLM-assisted) labeled rows: {len(new_df)}")
    print(f"written to: {OUT_CSV}")

    compare_reports(old_df, new_df)

    section("REMINDER")
    print(
        "data/processed/intent_training_sample_silver.csv contains LLM-ASSISTED SILVER "
        "LABELS for training only. These are NOT human-labeled, and NOT the 150-250 "
        "example hand-labeled golden evaluation set the assignment requires."
    )


if __name__ == "__main__":
    main()
