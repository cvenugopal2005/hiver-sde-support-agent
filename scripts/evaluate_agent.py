"""Step 16: evaluation harness for the full pipeline (intent classifier ->
retrieval -> MOCK-mode reply generation -> AUTO-HANDLE/ESCALATE decision)
against data/processed/golden_eval.csv.

*** GOLDEN SET PROVENANCE WARNING (see docs/project_notes.txt, Step 15 ***
*** addendum #2) *** -- of the 200 rows in golden_eval.csv, only 10 were
genuinely hand-labeled by a person. The other 190 were LLM-generated
proposals that were bulk auto-approved WITHOUT individual human review, on
explicit instruction, to finish the set quickly. This file is NOT a fully
hand-labelled ground truth set and every metric below inherits that
limitation -- they measure agreement with a mostly-LLM-labeled reference,
not verified real-world correctness. This warning is also printed at the
top of every run of this script and stored in evaluation_results.json.

METHOD NOTES (read before trusting a number):
- Input text: for each golden row, only the FIRST customer message is
  extracted from conversation_context (the same "a new customer message
  arrives" scenario generate_support_reply() and decide_handling() were
  built and demoed for in Steps 12/13) -- not the full multi-turn
  transcript. The intent classifier was actually trained on full
  conversation_context strings (Step 10), so this evaluation measures the
  pipeline as it is actually used, inheriting the same train/inference
  mismatch already flagged in Step 12, not a new one introduced here.
- Reply generation always runs in MOCK mode (LLM_PROVIDER is explicitly
  cleared before running), per the assignment's instruction for this step.
- predicted_response_behavior is derived from the TOP retrieved historical
  example's response_type (Step 11), mapped into the golden set's 6-value
  vocabulary. That mapping is LOSSY in one specific way: Step 11's
  "other/unclear" response_type merges what were originally two different
  categories (an escalation referral, and a genuinely unclear reply) into
  one, so it is mapped here to "unclear". Any golden row labeled
  "escalation" can therefore never be matched by this mapping -- a real,
  disclosed limitation of comparing across two different vocabularies, not
  a scoring bug.
- Golden rows labeled intent=UNCLEAR can never be predicted correctly
  either -- the classifier's classes never include UNCLEAR (Step 10
  excluded UNCLEAR from training). Included in the metrics anyway, for
  honesty, rather than quietly dropped.

Outputs:
    data/processed/evaluation_results.json      -- all aggregate metrics
    data/processed/evaluation_predictions.csv   -- one row per golden example

Usage:
    python scripts/evaluate_agent.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Force mock mode regardless of any stray environment configuration -- this
# harness must be reproducible without an API key, per the assignment.
os.environ.pop("LLM_PROVIDER", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide_handling import decide_handling  # noqa: E402
from generate_support_reply import generate_support_reply  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
GOLDEN_PATH = DATA_DIR / "golden_eval.csv"
TRAINING_SILVER_PATH = DATA_DIR / "intent_training_sample_silver.csv"
RESULTS_PATH = DATA_DIR / "evaluation_results.json"
PREDICTIONS_PATH = DATA_DIR / "evaluation_predictions.csv"

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
INTENT_LABELS_FOR_SCORING = INTENTS + ["UNCLEAR"]  # golden can contain UNCLEAR; classifier never predicts it
HANDLING_OPTIONS = ["AUTO-HANDLE", "ESCALATE"]

# Step 11's response_type vocabulary -> the golden set's expected_response_behavior
# vocabulary. LOSSY: see the module docstring's note on "other/unclear".
RESPONSE_TYPE_TO_BEHAVIOR = {
    "direct instruction/link": "direct_instruction",
    "clarifying question": "clarifying_question",
    "DM handoff": "dm_handoff",
    "confirmation/status": "confirmation/status",
    "other/unclear": "unclear",
}

N_FAILURE_EXAMPLES = 10


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_provenance_warning() -> None:
    print(
        "\n*** GOLDEN SET WARNING: golden_eval.csv is 10 human-labelled + 190 "
        "LLM-generated (bulk auto-approved, not individually human-reviewed) "
        "examples. It is NOT a fully hand-labelled ground truth set. Every "
        "metric below measures agreement with that reference, not verified "
        "real-world correctness. See docs/project_notes.txt. ***\n"
    )


def extract_first_customer_message(conversation_context: str) -> str:
    for line in conversation_context.splitlines():
        if line.startswith("CUSTOMER:"):
            return line[len("CUSTOMER:"):].strip()
    return conversation_context.strip()  # defensive fallback, should not trigger given the data's format


def get_majority_intent_baseline() -> str:
    """Reuses the same source/method as Step 10's DummyClassifier(strategy=
    'most_frequent'): the mode of the TRAINING labels (UNCLEAR excluded),
    not the golden set's own labels -- a majority baseline must not peek at
    the evaluation labels it's being compared against."""
    df = pd.read_csv(TRAINING_SILVER_PATH, dtype=str)
    df = df[df["intent"] != "UNCLEAR"]
    return df["intent"].value_counts().idxmax()


def run_pipeline_on_golden_set(golden_df: pd.DataFrame, majority_intent: str) -> pd.DataFrame:
    rows = []
    for i, row in golden_df.iterrows():
        customer_message = extract_first_customer_message(row["conversation_context"])

        result = generate_support_reply(customer_message)
        assert result["mode"] == "mock", f"expected mock mode, got {result['mode']!r} for tweet_id {row['tweet_id']}"

        predicted_intent = result["predicted_intent"]
        retrieved = result["retrieved_examples"]
        draft_reply = result["draft_reply"]
        top_response_type = retrieved[0]["response_type"] if retrieved else None
        predicted_response_behavior = RESPONSE_TYPE_TO_BEHAVIOR.get(top_response_type, "unclear")

        decision = decide_handling(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_evidence=retrieved,
            draft_reply=draft_reply,
        )
        predicted_handling = decision["decision"]

        rows.append(
            {
                "tweet_id": row["tweet_id"],
                "customer_message": customer_message,
                "golden_intent": row["intent"],
                "predicted_intent": predicted_intent,
                "intent_correct": predicted_intent == row["intent"],
                "golden_expected_handling": row["expected_handling"],
                "predicted_handling": predicted_handling,
                "handling_correct": predicted_handling == row["expected_handling"],
                "handling_reason": decision["reason"],
                "golden_expected_response_behavior": row["expected_response_behavior"],
                "predicted_response_behavior": predicted_response_behavior,
                "response_behavior_correct": predicted_response_behavior == row["expected_response_behavior"],
                "draft_reply": draft_reply,
                "top_retrieval_similarity": retrieved[0]["similarity_score"] if retrieved else None,
                "majority_intent_baseline": majority_intent,
                "majority_intent_baseline_correct": majority_intent == row["intent"],
                "always_escalate_baseline_correct": row["expected_handling"] == "ESCALATE",
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  ...processed {i + 1}/{len(golden_df)}")
    return pd.DataFrame(rows)


def intent_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=INTENT_LABELS_FOR_SCORING, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=INTENT_LABELS_FOR_SCORING, zero_division=0),
        "per_class": classification_report(
            y_true, y_pred, labels=INTENT_LABELS_FOR_SCORING, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=INTENT_LABELS_FOR_SCORING).tolist(),
        "confusion_matrix_labels": INTENT_LABELS_FOR_SCORING,
    }


def handling_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=HANDLING_OPTIONS, pos_label="AUTO-HANDLE", average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auto_handle_precision": precision,
        "auto_handle_recall": recall,
        "auto_handle_f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=HANDLING_OPTIONS).tolist(),
        "confusion_matrix_labels": HANDLING_OPTIONS,
    }


def main() -> None:
    section("GOLDEN SET")
    golden_df = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False)
    print(f"rows: {len(golden_df)}")
    print_provenance_warning()

    majority_intent = get_majority_intent_baseline()
    print(f"majority intent baseline (from training labels, not golden labels): {majority_intent!r}")

    section("RUNNING FULL PIPELINE ON ALL GOLDEN EXAMPLES (mock mode)")
    preds_df = run_pipeline_on_golden_set(golden_df, majority_intent)

    section("INTENT METRICS -- SYSTEM")
    system_intent = intent_metrics(preds_df["golden_intent"], preds_df["predicted_intent"])
    print(f"accuracy={system_intent['accuracy']:.3f}  macro_f1={system_intent['macro_f1']:.3f}  "
          f"weighted_f1={system_intent['weighted_f1']:.3f}")

    section("INTENT METRICS -- MAJORITY-CLASS BASELINE")
    baseline_intent = intent_metrics(preds_df["golden_intent"], preds_df["majority_intent_baseline"])
    print(f"accuracy={baseline_intent['accuracy']:.3f}  macro_f1={baseline_intent['macro_f1']:.3f}  "
          f"weighted_f1={baseline_intent['weighted_f1']:.3f}")

    section("HANDLING METRICS -- SYSTEM")
    system_handling = handling_metrics(preds_df["golden_expected_handling"], preds_df["predicted_handling"])
    print(f"accuracy={system_handling['accuracy']:.3f}  AUTO-HANDLE precision={system_handling['auto_handle_precision']:.3f}  "
          f"recall={system_handling['auto_handle_recall']:.3f}  f1={system_handling['auto_handle_f1']:.3f}")

    section("HANDLING METRICS -- ALWAYS-ESCALATE BASELINE")
    always_escalate_preds = pd.Series(["ESCALATE"] * len(preds_df))
    baseline_handling = handling_metrics(preds_df["golden_expected_handling"], always_escalate_preds)
    print(f"accuracy={baseline_handling['accuracy']:.3f}  AUTO-HANDLE precision={baseline_handling['auto_handle_precision']:.3f}  "
          f"recall={baseline_handling['auto_handle_recall']:.3f}  f1={baseline_handling['auto_handle_f1']:.3f}")

    section("RESPONSE-BEHAVIOR ACCURACY -- SYSTEM")
    behavior_accuracy = accuracy_score(preds_df["golden_expected_response_behavior"], preds_df["predicted_response_behavior"])
    print(f"accuracy={behavior_accuracy:.3f}")
    print("(no baseline requested for this metric)")

    section("PER-INTENT METRICS (SYSTEM)")
    per_class_df = pd.DataFrame(system_intent["per_class"]).T
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(per_class_df.round(3).to_string())

    section("CONFUSION MATRIX -- INTENT (rows=golden, cols=predicted)")
    print("labels order:", INTENT_LABELS_FOR_SCORING)
    print(np.array(system_intent["confusion_matrix"]))

    section("CONFUSION MATRIX -- HANDLING (rows=golden, cols=predicted)")
    print("labels order:", HANDLING_OPTIONS)
    print(np.array(system_handling["confusion_matrix"]))

    section(f"{N_FAILURE_EXAMPLES} ACTUAL FAILURE EXAMPLES (first N in file order where intent or handling is wrong)")
    failures = preds_df[~(preds_df["intent_correct"] & preds_df["handling_correct"])].head(N_FAILURE_EXAMPLES)
    failure_examples = []
    for _, row in failures.iterrows():
        print(f"\ntweet_id={row['tweet_id']}")
        print(f"  customer_message: {row['customer_message'][:150]}")
        print(f"  intent:    golden={row['golden_intent']!r:45s} predicted={row['predicted_intent']!r} "
              f"({'OK' if row['intent_correct'] else 'WRONG'})")
        print(f"  handling:  golden={row['golden_expected_handling']!r:12s} predicted={row['predicted_handling']!r} "
              f"({'OK' if row['handling_correct'] else 'WRONG'}) -- {row['handling_reason']}")
        print(f"  draft_reply: {row['draft_reply'][:150]}")
        failure_examples.append(row.to_dict())

    section("SAVING OUTPUTS")
    preds_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"predictions saved to: {PREDICTIONS_PATH}")

    results = {
        "golden_set_provenance": {
            "total_rows": len(golden_df),
            "human_labelled_rows": 10,
            "llm_generated_bulk_approved_rows": 190,
            "warning": (
                "This is NOT a fully hand-labelled ground truth evaluation set. "
                "190/200 labels were LLM-generated and bulk auto-approved without "
                "individual human review. All metrics below measure agreement with "
                "this reference, not verified real-world correctness."
            ),
        },
        "method_notes": {
            "input_text": "first CUSTOMER message extracted from conversation_context, not the full transcript",
            "reply_mode": "mock (no LLM API key configured; LLM_PROVIDER cleared before running)",
            "response_behavior_mapping": RESPONSE_TYPE_TO_BEHAVIOR,
            "response_behavior_mapping_limitation": (
                "Step 11's 'other/unclear' response_type merges what were originally an "
                "escalation referral and a genuinely unclear reply into one bucket, mapped "
                "here to 'unclear' -- golden rows labeled 'escalation' can never be matched."
            ),
        },
        "majority_intent_baseline_value": majority_intent,
        "intent": {"system": system_intent, "majority_baseline": baseline_intent},
        "handling": {"system": system_handling, "always_escalate_baseline": baseline_handling},
        "response_behavior": {"system_accuracy": behavior_accuracy},
        "failure_examples": failure_examples,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"results saved to: {RESULTS_PATH}")

    section("SUMMARY: SYSTEM vs BASELINES")
    print(f"{'metric':30s} {'system':>10s} {'baseline':>10s}")
    print(f"{'intent accuracy':30s} {system_intent['accuracy']:>10.3f} {baseline_intent['accuracy']:>10.3f}")
    print(f"{'intent macro_f1':30s} {system_intent['macro_f1']:>10.3f} {baseline_intent['macro_f1']:>10.3f}")
    print(f"{'intent weighted_f1':30s} {system_intent['weighted_f1']:>10.3f} {baseline_intent['weighted_f1']:>10.3f}")
    print(f"{'handling accuracy':30s} {system_handling['accuracy']:>10.3f} {baseline_handling['accuracy']:>10.3f}")
    print(f"{'handling AUTO-HANDLE f1':30s} {system_handling['auto_handle_f1']:>10.3f} {baseline_handling['auto_handle_f1']:>10.3f}")
    print(f"{'response_behavior accuracy':30s} {behavior_accuracy:>10.3f} {'n/a':>10s}")

    print_provenance_warning()


if __name__ == "__main__":
    main()
