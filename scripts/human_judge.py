"""Step 18: measure LLM-judge vs human agreement.

Builds a stratified 30-example sample from data/processed/llm_judge_predictions.csv
(Step 17's judged replies, joined with Step 16's evaluation_predictions.csv
context), shows a human reviewer the same information the LLM judge saw
(customer message, generated reply, retrieved evidence, and the LLM judge's
4 scores), and asks the human to independently score the SAME 4 dimensions
(relevance, groundedness, helpfulness, safety; 1-5 each).

*** THE HUMAN SCORES MUST COME FROM AN ACTUAL PERSON TYPING THEM IN. ***
This script never generates, guesses, or fills in a human score itself --
run this interactively in a real terminal. Claude does not run this to
produce scores; doing so would defeat the entire point of measuring
human/LLM agreement.

Progress is saved after every fully-scored example, so it's safe to stop
and resume. The stratified sample itself is built once (deterministically,
seed below) and persisted to data/processed/human_judge_results.csv with
blank human_* columns -- re-running the script resumes from whatever is
already saved there rather than re-sampling.

Outputs:
    data/processed/human_judge_results.csv  -- 30 rows: context, LLM scores,
                                                human scores (filled in by a
                                                person), agreement is computed
                                                from this file via --report

Usage:
    python scripts/human_judge.py             # start/resume human scoring
    python scripts/human_judge.py --report    # agreement metrics only, no prompts
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_support_reply import load_retriever  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
JUDGED_PATH = DATA_DIR / "llm_judge_predictions.csv"
RESULTS_PATH = DATA_DIR / "human_judge_results.csv"

SEED = 8080  # distinct from every other seed used in this project
SAMPLE_SIZE = 30
DIMENSIONS = ["relevance", "groundedness", "helpfulness", "safety"]
TOP_K_EVIDENCE = 3
N_DISAGREEMENT_EXAMPLES = 5


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------


STRATA_COLUMNS = ["predicted_intent", "predicted_response_behavior", "predicted_handling"]


def build_stratified_sample(judged_df: pd.DataFrame) -> pd.DataFrame:
    """Two-phase stratification:

    Phase 1 (coverage): greedily picks rows so every distinct value of
    predicted_intent, predicted_response_behavior, and predicted_handling
    that occurs in the 200 judged examples is covered by at least one
    sampled row. This is NOT the same as covering every joint combination
    of the three (there are 37 of those among 200 rows -- more than the
    30-example budget allows, so guaranteeing one-of-each-joint-combo is
    infeasible). Covering the three dimensions' individual values (8 + 5 +
    2 = 15 values total) is what "covering different intents, response
    behaviors, AUTO-HANDLE/ESCALATE outcomes" can actually mean at n=30,
    and each row typically covers 3 required values at once, so this uses
    well under 30 rows in practice.

    Phase 2 (fill): remaining sample slots are filled proportionally from
    the leftover pool, stratified by the joint (intent, behavior, handling)
    combination, to round out the sample without just duplicating whatever
    Phase 1 already grabbed.
    """
    df = judged_df.copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    required_values = {(col, v) for col in STRATA_COLUMNS for v in df[col].unique()}
    pool = df.index.tolist()
    rng.shuffle(pool)

    chosen_idx: list[int] = []
    covered: set[tuple[str, str]] = set()
    while covered != required_values and pool:
        best_idx, best_new = None, -1
        for idx in pool:
            row = df.loc[idx]
            new_covers = sum(1 for col in STRATA_COLUMNS if (col, row[col]) not in covered)
            if new_covers > best_new:
                best_idx, best_new = idx, new_covers
        if best_new <= 0:
            break
        chosen_idx.append(best_idx)
        row = df.loc[best_idx]
        covered.update((col, row[col]) for col in STRATA_COLUMNS)
        pool.remove(best_idx)

    n_needed = SAMPLE_SIZE - len(chosen_idx)
    if n_needed > 0 and pool:
        rest = df.loc[pool].copy()
        rest["_stratum"] = rest[STRATA_COLUMNS].agg(" | ".join, axis=1)
        groups = {k: g for k, g in rest.groupby("_stratum")}

        alloc = {k: max(1, round(len(g) / len(rest) * n_needed)) for k, g in groups.items()}
        while sum(alloc.values()) > n_needed:
            candidates = [k for k in alloc if alloc[k] > 1]
            if not candidates:
                break
            k = max(candidates, key=lambda k: alloc[k])
            alloc[k] -= 1
        while sum(alloc.values()) < n_needed:
            k = max(groups, key=lambda k: len(groups[k]) - alloc.get(k, 0))
            alloc[k] = alloc.get(k, 0) + 1

        for stratum, n in alloc.items():
            group_idx = groups[stratum].index.to_numpy()
            n = min(n, len(group_idx))
            picked = rng.choice(group_idx, size=n, replace=False)
            chosen_idx.extend(int(i) for i in picked)

    chosen_idx = chosen_idx[:SAMPLE_SIZE]
    sample = df.loc[chosen_idx].reset_index(drop=True)
    # Deterministic final order (not grouped by stratum/coverage-order) so
    # the review doesn't march through one intent/behavior at a time.
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return sample


def fetch_evidence_summary(customer_message: str, predicted_intent: str) -> str:
    retriever = load_retriever()
    retrieved = retriever.retrieve(customer_message, top_k=TOP_K_EVIDENCE, intent=predicted_intent)
    if not retrieved:
        return "(no historical evidence retrieved)"
    lines = []
    for i, r in enumerate(retrieved, start=1):
        lines.append(
            f"[{i}] similarity={r['similarity_score']:.3f} response_type={r['response_type']}\n"
            f"    historical customer message: {r['historical_customer_text']}\n"
            f"    historical AppleSupport reply: {r['historical_support_text']}"
        )
    return "\n".join(lines)


def build_working_df() -> pd.DataFrame:
    if RESULTS_PATH.exists():
        print(f"Resuming from {RESULTS_PATH}")
        return pd.read_csv(RESULTS_PATH, dtype=str, keep_default_na=False)

    print(f"No existing {RESULTS_PATH} -- building a fresh stratified sample of {SAMPLE_SIZE} (seed={SEED}).")
    judged_df = pd.read_csv(JUDGED_PATH, dtype={"tweet_id": "int64"})
    sample = build_stratified_sample(judged_df)

    rows = []
    for _, row in sample.iterrows():
        evidence_summary = fetch_evidence_summary(row["customer_message"], row["predicted_intent"])
        rows.append(
            {
                "tweet_id": row["tweet_id"],
                "customer_message": row["customer_message"],
                "draft_reply": row["draft_reply"],
                "retrieved_evidence": evidence_summary,
                "predicted_intent": row["predicted_intent"],
                "predicted_response_behavior": row["predicted_response_behavior"],
                "predicted_handling": row["predicted_handling"],
                "llm_relevance": row["relevance"],
                "llm_groundedness": row["groundedness"],
                "llm_helpfulness": row["helpfulness"],
                "llm_safety": row["safety"],
                "human_relevance": "",
                "human_groundedness": "",
                "human_helpfulness": "",
                "human_safety": "",
                "human_notes": "",
            }
        )
    df = pd.DataFrame(rows)
    save(df)
    return df


def save(df: pd.DataFrame) -> None:
    df.to_csv(RESULTS_PATH, index=False)


def is_scored(row: pd.Series) -> bool:
    return all(str(row[f"human_{dim}"]).strip() != "" for dim in DIMENSIONS)


# --------------------------------------------------------------------------
# Interactive scoring
# --------------------------------------------------------------------------


def ask_score(dimension: str) -> str | None:
    """Loops until a valid 1-5 score, 's', or 'q'. Returns 'q'/'s' or the
    digit string."""
    while True:
        try:
            choice = input(f"{dimension} [1-5/s/q]: ").strip().lower()
        except EOFError:
            return "q"
        if choice in ("s", "q"):
            return choice
        if choice in {"1", "2", "3", "4", "5"}:
            return choice
        print("Not a valid choice -- enter 1-5, s to skip this example, or q to quit.")


def score_loop(df: pd.DataFrame) -> None:
    remaining_idx = df.index[~df.apply(is_scored, axis=1)].tolist()
    if not remaining_idx:
        print("Nothing left to score. Run with --report to see agreement metrics.")
        return

    print(f"\nScoring {len(remaining_idx)} of {len(df)} examples. For each dimension: 1-5, "
          "s = skip this example, q = save and quit.\n")

    for position, idx in enumerate(remaining_idx, start=1):
        row = df.loc[idx]
        print(f"\n{'-' * 78}")
        print(f"[{position}/{len(remaining_idx)}] tweet_id={row['tweet_id']}")
        print(f"CUSTOMER MESSAGE: {row['customer_message']}")
        print(f"\nRETRIEVED EVIDENCE:\n{row['retrieved_evidence']}")
        print(f"\nGENERATED REPLY: {row['draft_reply']}")
        print(f"\npredicted_intent={row['predicted_intent']}  "
              f"predicted_response_behavior={row['predicted_response_behavior']}  "
              f"predicted_handling={row['predicted_handling']}")
        print(f"\nLLM JUDGE SCORES: relevance={row['llm_relevance']} groundedness={row['llm_groundedness']} "
              f"helpfulness={row['llm_helpfulness']} safety={row['llm_safety']}")
        print("-" * 78)

        answers = {}
        skipped = False
        for dim in DIMENSIONS:
            choice = ask_score(dim.capitalize())
            if choice == "q":
                save(df)
                left = len(remaining_idx) - position + 1
                print(f"Saved progress to {RESULTS_PATH}. {left} example(s) still unscored.")
                return
            if choice == "s":
                skipped = True
                break
            answers[dim] = choice

        if skipped:
            continue

        try:
            note = input("Notes (optional, Enter to skip): ").strip()
        except EOFError:
            note = ""

        for dim in DIMENSIONS:
            df.loc[idx, f"human_{dim}"] = answers[dim]
        df.loc[idx, "human_notes"] = note
        save(df)  # progressive save after every fully-scored example

    save(df)
    print("\nAll examples scored. Run with --report to see agreement metrics.")


# --------------------------------------------------------------------------
# Agreement metrics
# --------------------------------------------------------------------------


def compute_agreement(df: pd.DataFrame) -> dict:
    scored = df[df.apply(is_scored, axis=1)].copy()
    for dim in DIMENSIONS:
        scored[f"human_{dim}"] = scored[f"human_{dim}"].astype(int)
        scored[f"llm_{dim}"] = scored[f"llm_{dim}"].astype(int)
        scored[f"_diff_{dim}"] = (scored[f"human_{dim}"] - scored[f"llm_{dim}"]).abs()
        scored[f"_total_diff"] = scored.get("_total_diff", 0) + scored[f"_diff_{dim}"]

    per_dimension = {}
    for dim in DIMENSIONS:
        human = scored[f"human_{dim}"]
        llm = scored[f"llm_{dim}"]
        exact_agreement = float((human == llm).mean()) if len(scored) else None
        mae = float((human - llm).abs().mean()) if len(scored) else None
        if len(scored) >= 2 and human.nunique() > 1 and llm.nunique() > 1:
            corr, pvalue = pearsonr(human, llm)
            pearson = {"r": float(corr), "p_value": float(pvalue)}
        else:
            pearson = {"r": None, "note": "not computable (need >=2 scored examples with variance in both series)"}
        per_dimension[dim] = {"exact_agreement_rate": exact_agreement, "mean_absolute_difference": mae, "pearson": pearson}

    return {"n_scored": len(scored), "per_dimension": per_dimension}, scored


def print_report(df: pd.DataFrame) -> None:
    total = len(df)
    n_scored = int(df.apply(is_scored, axis=1).sum())
    print(f"\nTotal sample: {total}")
    print(f"Human-scored: {n_scored}")
    print(f"Remaining: {total - n_scored}")

    if n_scored == 0:
        print("\nNo examples scored yet -- nothing to compute agreement on.")
        return

    agreement, scored = compute_agreement(df)
    section(f"AGREEMENT: HUMAN vs LLM JUDGE (n={agreement['n_scored']})")
    for dim in DIMENSIONS:
        m = agreement["per_dimension"][dim]
        pearson_str = (
            f"r={m['pearson']['r']:.3f} (p={m['pearson']['p_value']:.3f})"
            if m["pearson"]["r"] is not None
            else m["pearson"]["note"]
        )
        print(f"  {dim:15s}: exact_agreement={m['exact_agreement_rate']:.2f}  "
              f"MAE={m['mean_absolute_difference']:.2f}  pearson={pearson_str}")

    if len(scored) >= 1:
        section(f"{N_DISAGREEMENT_EXAMPLES} EXAMPLES WITH LARGEST DISAGREEMENT")
        worst = scored.sort_values("_total_diff", ascending=False).head(N_DISAGREEMENT_EXAMPLES)
        for _, row in worst.iterrows():
            print(f"\ntweet_id={row['tweet_id']}  total_abs_diff={row['_total_diff']}")
            print(f"  customer: {row['customer_message'][:130]}")
            print(f"  reply: {row['draft_reply'][:130]}")
            for dim in DIMENSIONS:
                print(f"  {dim:13s}: human={row[f'human_{dim}']}  llm={row[f'llm_{dim}']}  "
                      f"diff={row[f'_diff_{dim}']}")
            if row.get("human_notes"):
                print(f"  human_notes: {row['human_notes']}")


def main() -> None:
    df = build_working_df()
    if "--report" in sys.argv:
        print_report(df)
        return
    score_loop(df)
    print_report(df)


if __name__ == "__main__":
    main()
