"""Step 6: pick a manageable manual-labeling subset (~400-500 rows) out of
the 1,000-row data/processed/intent_labeling_sample.csv from Step 5.

Reproducible, stratified sampling only -- no intent labels are assigned here
(none exist yet; this script never writes to the `intent` column). Diversity
is preserved by stratifying on:
    - thread length bucket (short=2, medium=3-4, long=5+)
    - a rough issue-topic bucket, computed with the same keyword patterns
      already used (and already flagged as approximate, non-authoritative)
      in scripts/validate_applesupport.py

The issue-topic bucket is used ONLY to spread the sample across topics so
one theme doesn't dominate by chance -- it is not written to the output
file and must never be treated as a label.

Output columns (exactly): tweet_id, conversation_context, intent,
labeling_notes. `intent` and `labeling_notes` are left blank.

Usage:
    python scripts/prepare_training_sample.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 77
TARGET_FRACTION = 0.45  # ~450 of 1000, within the requested 400-500 range

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "intent_labeling_sample.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "intent_training_sample.csv"

# Same patterns as scripts/validate_applesupport.py -- used here only to
# spread the sample across topics, not to assign any final label.
ISSUE_PATTERNS = {
    "hardware_repair_warranty": r"warrant|repair|replace|genius bar|apple store|hardware",
    "account_purchase_billing": r"password|passcode|apple ?id|log ?in|verif|charged|refund|subscri|credit card|payment|billing|purchase",
    "music_sync_dataloss": r"apple music|itunes|playlist|\bsync\b|delet|disappear|lost my (?:photo|contact)",
    "connectivity": r"wifi|wi-fi|bluetooth|\blte\b|cellular|connectiv|hotspot|disconnect",
    "messaging_facetime": r"imessage|\bsms\b|facetime|text message|voicemail",
    "battery_performance": r"batter|charg|freez|crash|\blag\b|laggy|\bslow\b|hang|unresponsive|stuck",
    "software_bug_generic": r"keyboard|autocorrect|\bbug\b|glitch|question mark",
}
PRIORITY = list(ISSUE_PATTERNS.keys())  # same priority order used in Step 4's rule


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def thread_length_bucket(size: int) -> str:
    if size <= 2:
        return "short"
    if size <= 4:
        return "medium"
    return "long"


def rough_issue_bucket(text: str) -> str:
    low = str(text).lower()
    for name in PRIORITY:
        if re.search(ISSUE_PATTERNS[name], low):
            return name
    return "other_uncategorized"


def main() -> None:
    section("LOADING data/processed/intent_labeling_sample.csv (Step 5 output)")
    df = pd.read_csv(IN_PATH)
    print(f"rows available: {len(df)}")

    df["_length_bucket"] = df["thread_size"].apply(thread_length_bucket)
    df["_issue_bucket"] = df["customer_text"].apply(rough_issue_bucket)
    df["_stratum"] = df["_length_bucket"] + "|" + df["_issue_bucket"]

    section("STRATIFICATION (for sampling diversity only, not labels)")
    print(df["_length_bucket"].value_counts().to_string())
    print()
    print(df["_issue_bucket"].value_counts().to_string())

    rng = np.random.default_rng(SEED)

    def sample_group(g: pd.DataFrame) -> pd.DataFrame:
        n = max(1, round(len(g) * TARGET_FRACTION))
        n = min(n, len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        return g.loc[idx]

    sampled = (
        df.groupby("_stratum", group_keys=False)
        .apply(sample_group, include_groups=False)
        .reset_index(drop=True)
    )

    section("RESULT")
    print(f"rows selected: {len(sampled)} (target ~{round(len(df) * TARGET_FRACTION)}, "
          f"requested range 400-500)")

    out = sampled[["tweet_id", "conversation_context"]].copy()
    out["intent"] = ""
    out["labeling_notes"] = ""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"written to: {OUT_PATH}")
    print(f"empty intent count: {(out['intent'] == '').sum()} / {len(out)} (must equal total)")


if __name__ == "__main__":
    main()
