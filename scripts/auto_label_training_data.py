"""Deterministic, rule-based auto-labeling of the 450-row training sample.

IMPORTANT SCOPE NOTE: these are WEAK / SILVER labels for training an intent
classifier quickly. They are produced by a fixed set of explainable regex
rules derived directly from configs/intent_taxonomy.md and
configs/labeling_guide.md -- no LLM, no external API, no manual judgment.
They are explicitly NOT the 150-250-example hand-labeled golden evaluation
set the assignment separately requires; that set must still be labeled by
a human, independently of this script and of these training labels.

Method: for each row, search the FULL conversation_context (both CUSTOMER
and APPLESUPPORT lines) against the same priority-ordered rule set already
used (and already flagged as an approximate proxy, not ground truth) in
scripts/validate_applesupport.py and scripts/prepare_training_sample.py:

    0. "I" autocorrect glitch  -> always Software / App Bug Report
       (explicit taxonomy rule, checked first, overrides all other matches)
    1. Hardware, Repair & Warranty
    2. Account, Purchases & Billing
    3. Apple Music / iCloud Sync & Data Loss
    4. Connectivity
    5. Messaging & Communication
    6. Battery & Performance
    7. Software / App Bug Report (specific bug/glitch language)
    8. General / Pre-Purchase Inquiry (only if no complaint language at all)
    9. Software / App Bug Report (fallback: some complaint/malfunction
       language present but no specific subsystem named -- see taxonomy's
       "fallback when nothing specific is named" rule)
   10. UNCLEAR (nothing above matched)

Usage:
    python scripts/auto_label_training_data.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_applesupport import is_i_autocorrect_bug_message  # noqa: E402

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

# Priority-ordered rules (1-6 in the taxonomy's primary-intent rule).
# (intent, regex, note, confidence)
PRIORITY_RULES = [
    (
        "Hardware, Repair & Warranty",
        r"warrant|repair|replace|genius bar|apple store|hardware",
        "Hardware/warranty/repair language present (device defect, repair, or warranty/replacement request).",
        "high",
    ),
    (
        "Account, Purchases & Billing",
        r"password|passcode|apple ?id|log ?in|verif|charged|refund|subscri|credit card|payment|billing|"
        r"purchase history|unauthorized purchase|purchase (?:declined|failed)",
        "Account, login, or billing/purchase language present (Apple ID, password, charge, refund, or subscription). "
        "Note: bare mentions of 'purchase' (e.g. 'I purchased an iPhone last year') are deliberately NOT matched here "
        "-- only purchase-PROBLEM phrasing is, to avoid mislabeling background context as a billing issue.",
        "high",
    ),
    (
        "Apple Music / iCloud Sync & Data Loss",
        r"apple music|itunes|playlist|\bsync\b|"
        r"(?:photos?|contacts?|songs?|voicemails?|notes?)\W{0,20}(?:delet\w*|disappear\w*|gone|vanish\w*)|"
        r"(?:delet\w*|disappear\w*|vanish\w*)\W{0,20}(?:photos?|contacts?|songs?|voicemails?|notes?)",
        "Apple Music/iCloud sync or data-loss language present (songs, photos, contacts, or backups). "
        "Note: 'disappear'/'delete' alone are NOT matched -- only when near a specific content noun -- "
        "to avoid catching unrelated uses (e.g. a WiFi signal 'disappearing').",
        "high",
    ),
    (
        "Connectivity",
        r"wifi|wi-fi|bluetooth|\blte\b|cellular|connectiv|hotspot|disconnect",
        "Connectivity language present (WiFi, Bluetooth, or cellular).",
        "high",
    ),
    (
        "Messaging & Communication",
        r"imessage|\bsms\b|facetime|text message|voicemail",
        "Messaging/communication feature language present (iMessage, SMS, FaceTime, or voicemail).",
        "high",
    ),
    (
        "Battery & Performance",
        r"batter|charg|freez|crash|\blag\b|laggy|\bslow\b|hang|unresponsive|stuck",
        "Battery/performance problem language present (freezing, crashing, slow, or battery drain), often after an update.",
        "high",
    ),
]

SOFTWARE_BUG_SPECIFIC = r"keyboard|autocorrect|\bbug\b|glitch|question mark"
PRE_PURCHASE = (
    r"can i buy|before i buy|planning to buy|thinking about (?:buying|getting)|"
    r"is there a way to (?:buy|purchase|upgrade)|financing|applecare|what.?s the policy|"
    r"how (?:long|much).{0,25}(?:applecare|warrant|cost|price|financ)"
)
GENERIC_COMPLAINT = (
    r"\bfix\b|sucks|broken|doesn'?t work|not working|please help|\bissue\b|\bproblem\b|ruin|"
    r"annoying|\berror\b|\bwrong\b|failed|\bfail\b|\bstuck\b|worst|terrible|garbage|trash|useless|can'?t\b"
)


def classify(context: str) -> tuple[str, str, str, list[str]]:
    """Returns (intent, note, confidence, matched_priority_categories)."""
    low = str(context).lower()

    matched_priority = [name for name, pattern, _note, _conf in PRIORITY_RULES if re.search(pattern, low)]

    if is_i_autocorrect_bug_message(context):
        return (
            "Software / App Bug Report",
            "'I' autocorrect glitch (iOS 11.1) -- per taxonomy, always Software/App Bug Report regardless of other mentions.",
            "high",
            matched_priority,
        )

    if matched_priority:
        intent, _pattern, note, conf = next(r for r in PRIORITY_RULES if r[0] == matched_priority[0])
        return intent, note, conf, matched_priority

    if re.search(SOFTWARE_BUG_SPECIFIC, low):
        return (
            "Software / App Bug Report",
            "Software/app bug or glitch language present, not covered by a more specific category.",
            "medium",
            matched_priority,
        )

    has_complaint = bool(re.search(GENERIC_COMPLAINT, low))
    if re.search(PRE_PURCHASE, low) and not has_complaint:
        return (
            "General / Pre-Purchase Inquiry",
            "Pre-purchase or policy question language present, no malfunction described.",
            "medium",
            matched_priority,
        )

    if has_complaint:
        return (
            "Software / App Bug Report",
            "General malfunction/complaint language with no specific subsystem named; defaulted to "
            "Software/App Bug Report per the taxonomy's fallback rule.",
            "low",
            matched_priority,
        )

    return (
        "UNCLEAR",
        "No identifiable problem, request, or specific topic found in the visible conversation text.",
        "none",
        matched_priority,
    )


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    section("LOADING data/processed/intent_training_sample.csv")
    df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)
    print(f"rows: {len(df)}")

    results = df["conversation_context"].apply(classify)
    df["intent"] = results.apply(lambda r: r[0])
    df["labeling_notes"] = results.apply(lambda r: r[1])
    confidence = results.apply(lambda r: r[2])  # internal only, not written to the output CSV
    multi_match = results.apply(lambda r: len(r[3]) > 1)  # internal only

    out = df[["tweet_id", "conversation_context", "intent", "labeling_notes"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    section("REPORT: TOTALS")
    total = len(df)
    print(f"total examples: {total}")
    counts = df["intent"].value_counts()
    print("\ncount and percentage per intent:")
    for intent in INTENTS:
        n = int(counts.get(intent, 0))
        print(f"  {intent}: {n} ({n / total * 100:.1f}%)")
    n_unclear = int(counts.get("UNCLEAR", 0))
    print(f"  UNCLEAR: {n_unclear} ({n_unclear / total * 100:.1f}%)")

    section("REPORT: CONFIDENCE BREAKDOWN (internal only, not in output CSV)")
    print(confidence.value_counts().to_string())

    section("REPORT: LOW-COUNT CATEGORIES (< 5% of the 450 rows)")
    low_count = [(i, int(counts.get(i, 0))) for i in INTENTS if counts.get(i, 0) < 0.05 * total]
    if low_count:
        for name, n in low_count:
            print(f"  {name}: {n} ({n / total * 100:.1f}%)")
    else:
        print("  none")

    section("REPORT: ONE EXAMPLE PER INTENT (real rows)")
    for intent in INTENTS + ["UNCLEAR"]:
        subset = df[df["intent"] == intent]
        if len(subset) == 0:
            print(f"\n{intent}: (no examples)")
            continue
        row = subset.iloc[0]
        first_line = row["conversation_context"].splitlines()[0]
        print(f"\n{intent} (tweet_id={row['tweet_id']}):")
        print(f"  {first_line[:150]}")
        print(f"  note: {row['labeling_notes']}")

    section("REPORT: OBVIOUS AMBIGUOUS CASES (matched more than one priority category)")
    ambiguous = df[multi_match]
    print(f"count: {len(ambiguous)} / {total} ({len(ambiguous) / total * 100:.1f}%)")
    for _, row in ambiguous.head(5).iterrows():
        first_line = row["conversation_context"].splitlines()[0]
        print(f"  tweet_id={row['tweet_id']}: assigned '{row['intent']}' -- {first_line[:120]}")

    print(f"\nwritten to: {OUT_PATH}")


if __name__ == "__main__":
    main()
