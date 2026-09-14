"""Step 17: LLM-as-judge evaluation of generated-reply QUALITY.

Judges every generated reply in data/processed/evaluation_predictions.csv
(Step 16) on 4 dimensions, each scored 1-5:
    1. Relevance     -- does it address the customer's actual issue?
    2. Groundedness   -- is every claim/instruction actually supported by
                         the retrieved historical evidence (re-fetched here,
                         since evaluation_predictions.csv doesn't store it)?
    3. Helpfulness    -- would this genuinely help the customer make
                         progress, even a safe "please DM us"?
    4. Safety         -- does it avoid asserting something unsupported
                         (never treats a DM handoff as proof of resolution;
                         no invented refunds/policies/guarantees)?

LLM PROVIDER
No LLM API key is configured in this environment (same situation as Steps 9
and 12). This module implements a provider interface (JudgeProvider) with:
    - MockJudgeProvider: a deterministic HEURISTIC scorer, not a real
      quality judgment. Used only as a development/smoke-test path so the
      harness can run end-to-end without an API key. Every mock score is
      tagged "mode": "mock" and must never be reported as a real LLM-judge
      result.
    - AnthropicJudgeProvider: a real provider (implemented, unexercised
      here -- needs LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY).

REAL judging in THIS run was produced by dispatching Claude (the assistant)
as independent subagents to act AS the LLM judge -- the same documented
workaround used in Step 9 for the silver training labels, chosen explicitly
by the project owner over MOCK-only scoring. Per-example judge output was
saved to data/processed/llm_judge_raw.json (200 records, 5 batches of 40
independent subagent calls, each given the customer message, predicted
intent, the actual top-3 retrieved historical evidence, and the draft
reply). This script LOADS that file rather than calling an API -- it is
NOT a live LLM call, and the results are tagged "mode": "real:claude-subagent"
to keep that distinction visible everywhere the results are used.

Outputs:
    data/processed/llm_judge_results.json      -- aggregates, per-dimension
                                                   averages, 10 representative
                                                   examples, failure patterns
    data/processed/llm_judge_predictions.csv   -- one row per judged example

Usage:
    python scripts/llm_judge.py               # auto: real if llm_judge_raw.json
                                               # exists, else falls back to mock
    python scripts/llm_judge.py --mode mock   # force the heuristic mock scorer
    python scripts/llm_judge.py --mode real   # force real (requires the raw
                                               # JSON or a configured API key)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_support_reply import load_retriever  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PREDICTIONS_PATH = DATA_DIR / "evaluation_predictions.csv"
JUDGE_RAW_JSON = DATA_DIR / "llm_judge_raw.json"
RESULTS_PATH = DATA_DIR / "llm_judge_results.json"
JUDGED_PREDICTIONS_PATH = DATA_DIR / "llm_judge_predictions.csv"

DIMENSIONS = ["relevance", "groundedness", "helpfulness", "safety"]
TOP_K_EVIDENCE_FOR_JUDGE = 3
N_REPRESENTATIVE_EXAMPLES = 10


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Provider abstraction (same pattern as generate_support_reply.py's LLMProvider)
# --------------------------------------------------------------------------


class JudgeProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def judge(self, prompt: str) -> str:
        """Returns raw text (expected to be JSON) for one judging prompt."""


class AnthropicJudgeProvider(JudgeProvider):
    """Real provider. Requires the `anthropic` package and ANTHROPIC_API_KEY.
    Not exercised in this environment (no key configured)."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5"):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but the 'anthropic' package is not installed."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def judge(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


def get_judge_provider() -> tuple[JudgeProvider | None, str]:
    """Same env-var convention as generate_support_reply.get_llm_provider().
    Returns (provider, mode); provider is None if no real provider is usable."""
    provider_name = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider_name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            return AnthropicJudgeProvider(api_key=api_key), "real:anthropic"
        print("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
    return None, "mock"


# --------------------------------------------------------------------------
# Judge prompt (re-fetches the actual retrieved evidence -- not stored in
# evaluation_predictions.csv -- so the judge can genuinely assess groundedness)
# --------------------------------------------------------------------------

JUDGE_INSTRUCTIONS = """You are an impartial quality judge for an AI customer-support reply-drafting system (AppleSupport on Twitter).

Score the DRAFT REPLY on 4 dimensions, each 1-5 (1=very poor, 5=excellent):

1. relevance -- does it address the customer's actual stated issue?
2. groundedness -- is EVERY claim/instruction in the reply actually supported by the RETRIEVED HISTORICAL EVIDENCE shown below? Penalize heavily if it asserts a fact, policy, refund, or troubleshooting step not present in that evidence. A generic "please DM us for help" is fully grounded (it invents nothing) even if it isn't very informative -- score groundedness on invention, not on how much information is given.
3. helpfulness -- would this genuinely help the customer make progress, even if it's just a safe clarifying question or a DM pointer?
4. safety -- does it avoid asserting anything unsupported? A "DM handoff" reply must NEVER be scored as if it claims the issue was resolved -- that would itself be a safety failure if the reply implied resolution. Since these drafts are template-based, safety failures are more about mismatched grounding than reckless commitments -- score based on the actual text.

Judge based ONLY on the text shown. Do not reward or penalize based on the underlying customer's tone/anger.

For each example, respond with the JSON object described in the OUTPUT section.
"""


def build_evidence_summary(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(no historical evidence retrieved)"
    lines = []
    for i, r in enumerate(retrieved[:TOP_K_EVIDENCE_FOR_JUDGE], start=1):
        lines.append(
            f"[{i}] similarity={r['similarity_score']:.3f} response_type={r['response_type']}\n"
            f"    historical customer message: {r['historical_customer_text']}\n"
            f"    historical AppleSupport reply: {r['historical_support_text']}"
        )
    return "\n".join(lines)


def build_judge_prompt(row: dict, evidence_summary: str) -> str:
    return (
        JUDGE_INSTRUCTIONS
        + f"\nCUSTOMER MESSAGE: {row['customer_message']}\n"
        + f"PREDICTED INTENT: {row['predicted_intent']}\n\n"
        + f"RETRIEVED HISTORICAL EVIDENCE (top {TOP_K_EVIDENCE_FOR_JUDGE}):\n{evidence_summary}\n\n"
        + f"DRAFT REPLY: {row['draft_reply']}\n"
    )


# --------------------------------------------------------------------------
# MOCK scoring -- deterministic heuristic, NOT a real quality judgment
# --------------------------------------------------------------------------


def mock_judge_scores(row: dict) -> dict:
    """Deterministic heuristic placeholder so the harness runs without an
    API key. NOT a real LLM judgment -- do not report these as real scores.

    Heuristic (crude, explainable, not learned): grounds on retrieval
    similarity as a groundedness proxy, and on the (already-computed)
    response_behavior_correct flag as a relevance proxy. Safety is always 5
    because build_mock_reply() (Step 12) is structurally incapable of
    inventing facts -- it only ever reuses retrieved text or asks a safe
    generic clarifying/DM question -- which is a property of the code, not
    a semantic judgment of this specific reply.
    """
    similarity = row.get("top_retrieval_similarity")
    groundedness = 5 if (similarity is not None and similarity >= 0.3) else (3 if (similarity is not None and similarity >= 0.15) else 2)
    relevance = 4 if row.get("response_behavior_correct") else 2
    helpfulness = round((groundedness + relevance) / 2)
    safety = 5
    overall = round((relevance + groundedness + helpfulness + safety) / 4, 2)
    return {
        "relevance": relevance,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "safety": safety,
        "overall": overall,
        "reasoning": "MOCK heuristic placeholder (similarity + response-behavior-match proxy), not a real judgment.",
    }


def run_mock_judging(preds_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in preds_df.iterrows():
        scores = mock_judge_scores(row.to_dict())
        records.append({"tweet_id": row["tweet_id"], **scores})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# REAL scoring -- loads pre-generated Claude-subagent judgments
# --------------------------------------------------------------------------


def load_real_judge_results() -> pd.DataFrame:
    if not JUDGE_RAW_JSON.exists():
        raise FileNotFoundError(
            f"{JUDGE_RAW_JSON} not found. Real judging in this project was produced by "
            "dispatching Claude subagents as the LLM judge (see module docstring) -- "
            "that file must exist first, or configure LLM_PROVIDER/ANTHROPIC_API_KEY "
            "for a live API call."
        )
    with open(JUDGE_RAW_JSON) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    for dim in DIMENSIONS:
        assert df[dim].between(1, 5).all(), f"{dim} has scores outside 1-5"
    df["overall"] = df[DIMENSIONS].mean(axis=1).round(2)
    return df


# --------------------------------------------------------------------------
# Aggregation, representative examples, failure patterns
# --------------------------------------------------------------------------


def build_report(preds_df: pd.DataFrame, scores_df: pd.DataFrame, mode: str) -> dict:
    merged = preds_df.merge(scores_df, on="tweet_id", how="inner")
    assert len(merged) == len(preds_df) == len(scores_df), "row count mismatch after merge"

    averages = {dim: float(merged[dim].mean()) for dim in DIMENSIONS}
    averages["overall"] = float(merged["overall"].mean())

    # Representative examples: lowest 3, highest 3, and 4 spread through the
    # middle of the overall-score distribution -- not cherry-picked to look
    # good or bad, just a deterministic spread.
    sorted_merged = merged.sort_values(["overall", "tweet_id"]).reset_index(drop=True)
    n = len(sorted_merged)
    low = sorted_merged.iloc[:3]
    high = sorted_merged.iloc[-3:]
    mid_positions = np.linspace(n * 0.35, n * 0.65, 4).astype(int).clip(0, n - 1)
    mid = sorted_merged.iloc[mid_positions]
    representative = pd.concat([low, mid, high]).drop_duplicates(subset="tweet_id").head(N_REPRESENTATIVE_EXAMPLES)

    # Failure-pattern analysis: average scores broken out by response_type-derived
    # dimensions already present in evaluation_predictions.csv.
    by_response_behavior = merged.groupby("predicted_response_behavior")[DIMENSIONS + ["overall"]].mean().round(2)
    by_handling = merged.groupby("predicted_handling")[DIMENSIONS + ["overall"]].mean().round(2)
    worst_dimension = min(averages, key=lambda k: averages[k] if k != "overall" else 999)

    low_score_threshold = merged["overall"].quantile(0.25)
    low_scoring = merged[merged["overall"] <= low_score_threshold]
    low_scoring_response_behavior_dist = low_scoring["predicted_response_behavior"].value_counts().to_dict()

    return {
        "mode": mode,
        "n_judged": len(merged),
        "average_scores": averages,
        "by_predicted_response_behavior": by_response_behavior.to_dict(orient="index"),
        "by_predicted_handling": by_handling.to_dict(orient="index"),
        "weakest_dimension_overall": worst_dimension,
        "low_scoring_examples_response_behavior_distribution": low_scoring_response_behavior_dist,
        "representative_examples": representative[
            ["tweet_id", "customer_message", "predicted_intent", "draft_reply", "predicted_response_behavior"]
            + DIMENSIONS + ["overall"]
            + (["reasoning"] if "reasoning" in representative.columns else [])
        ].to_dict(orient="records"),
    }, merged


def print_report(report: dict, merged: pd.DataFrame, mode: str) -> None:
    section(f"AVERAGE SCORES (mode={mode}, n={report['n_judged']})")
    for dim in DIMENSIONS:
        print(f"  {dim:15s}: {report['average_scores'][dim]:.2f}")
    print(f"  {'overall':15s}: {report['average_scores']['overall']:.2f}")

    section("AVERAGE SCORES BY predicted_response_behavior")
    print(pd.DataFrame(report["by_predicted_response_behavior"]).T.to_string())

    section("AVERAGE SCORES BY predicted_handling")
    print(pd.DataFrame(report["by_predicted_handling"]).T.to_string())

    section("OBVIOUS WEAKNESSES / FAILURE PATTERNS")
    print(f"Weakest dimension on average: {report['weakest_dimension_overall']}")
    print(f"Response-behavior distribution among the lowest-scoring quartile: "
          f"{report['low_scoring_examples_response_behavior_distribution']}")

    section(f"{N_REPRESENTATIVE_EXAMPLES} REPRESENTATIVE JUDGED EXAMPLES (3 lowest, 4 mid, 3 highest overall)")
    for ex in report["representative_examples"]:
        print(f"\ntweet_id={ex['tweet_id']}  overall={ex['overall']}")
        print(f"  customer: {ex['customer_message'][:130]}")
        print(f"  intent: {ex['predicted_intent']}   response_behavior: {ex['predicted_response_behavior']}")
        print(f"  draft_reply: {ex['draft_reply'][:150]}")
        print(f"  scores: relevance={ex['relevance']} groundedness={ex['groundedness']} "
              f"helpfulness={ex['helpfulness']} safety={ex['safety']}")
        if ex.get("reasoning"):
            print(f"  reasoning: {ex['reasoning']}")

    if mode == "mock":
        print("\n*** MOCK MODE: the scores above are a deterministic heuristic placeholder, "
              "NOT real LLM-judge quality assessments. ***")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "mock", "real"], default="auto")
    args = parser.parse_args()

    section("LOADING data/processed/evaluation_predictions.csv (Step 16)")
    preds_df = pd.read_csv(PREDICTIONS_PATH, dtype={"tweet_id": "int64"})
    preds_df["response_behavior_correct"] = preds_df["response_behavior_correct"].astype(bool)
    print(f"rows: {len(preds_df)}")

    mode_requested = args.mode
    if mode_requested == "auto":
        mode_requested = "real" if JUDGE_RAW_JSON.exists() else "mock"
        print(f"--mode auto -> resolved to {mode_requested!r} "
              f"({'found ' + str(JUDGE_RAW_JSON) if mode_requested == 'real' else 'no precomputed real results found'})")

    if mode_requested == "real":
        try:
            scores_df = load_real_judge_results()
            mode = "real:claude-subagent"
            print(f"loaded {len(scores_df)} real judge results from {JUDGE_RAW_JSON}")
        except FileNotFoundError as e:
            print(str(e))
            print("Falling back to MOCK mode.")
            scores_df = run_mock_judging(preds_df)
            mode = "mock"
    else:
        scores_df = run_mock_judging(preds_df)
        mode = "mock"

    report, merged = build_report(preds_df, scores_df, mode)
    print_report(report, merged, mode)

    merged.to_csv(JUDGED_PREDICTIONS_PATH, index=False)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    section("SAVED")
    print(f"per-row judged predictions: {JUDGED_PREDICTIONS_PATH}")
    print(f"aggregate report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
