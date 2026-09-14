# Hiver SDE Intern — AI Support Agent

## Project Overview

An AI customer-support agent for **AppleSupport** (Twitter), built from the
["Customer Support on Twitter" Kaggle dataset](data/raw/twcs.csv). Given a
new customer message, the pipeline:

1. **Classifies intent** (8 categories, `configs/intent_taxonomy.md`) via a
   TF-IDF + Logistic Regression classifier.
2. **Retrieves relevant historical AppleSupport replies** (TF-IDF cosine
   similarity over 80,639 real threads), tagging each with a
   `response_type` so a DM handoff is never mistaken for a resolution.
3. **Drafts a grounded reply** — currently in **mock/template mode** (no
   LLM API key is configured in this environment); the reply is built only
   from retrieved historical text, never invented.
4. **Decides AUTO-HANDLE vs ESCALATE** using 9 explicit, auditable,
   non-LLM rules.

It is then evaluated against a 200-example golden set, an LLM-as-judge
reply-quality harness, and (partially) a human-vs-judge agreement check.
See [reports/final_report.md](reports/final_report.md) for full results,
failure analysis, and honest limitations, and
[docs/project_notes.txt](docs/project_notes.txt) for a step-by-step build
log written for interview prep.

**Read this before trusting any number in this README:** the 200-row
evaluation set is **10 human-labelled + 190 LLM-generated** examples (see
Limitations below) — it is explicitly **not** a fully hand-labelled golden
set, and every metric below inherits that limitation.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Place the Kaggle dataset at `data/raw/twcs.csv` (not committed — see
`.gitignore`).

## Run Commands

Run in this order from a fresh clone (each step's output feeds the next;
`data/processed/*` is gitignored, so these must be (re)generated locally —
see **Reproduction Instructions** below):

```bash
# 1. Dataset & brand investigation (read-only, no artifacts required downstream)
python scripts/inspect_dataset.py
python scripts/analyze_brands.py

# 2. AppleSupport exploration (produces sample files used only for manual analysis)
python scripts/explore_applesupport.py
python scripts/validate_applesupport.py

# 3. Training data preparation
python scripts/prepare_intent_data.py          # 1000-thread labeling pool
python scripts/prepare_training_sample.py      # 450-example stratified subset
python scripts/auto_label_training_data.py     # deterministic rule-based labels (superseded by step 4)
python scripts/refine_training_labels.py       # merges in LLM-assisted silver labels (data/processed/llm_classification_raw.json must exist)

# 4. Train the intent classifier
python scripts/train_intent_classifier.py

# 5. Retrieval, reply generation, decision logic
python scripts/retrieve_historical_replies.py
python scripts/demo_retrieval_queries.py
python scripts/generate_support_reply.py "My battery drains fast since the update"
python scripts/demo_support_agent.py
python scripts/decide_handling.py "My battery drains fast since the update"
python scripts/demo_handling_decisions.py

# 6. Golden evaluation set (200 examples; see Limitations for its real composition)
python scripts/prepare_golden_set.py
# human labeling of golden_eval.csv happens here in a real run (scripts/label_golden_set.py) --
# in this project's actual history, only 10/200 rows were ever hand-labeled this way.
python scripts/bulk_approve_golden_labels.py   # only if choosing to skip full manual review, as this project did

# 7. Evaluation harness + LLM-as-judge + human agreement
python scripts/evaluate_agent.py
python scripts/llm_judge.py                    # loads data/processed/llm_judge_raw.json if present, else runs mock mode
python scripts/human_judge.py                  # MUST be run interactively by an actual person
python scripts/human_judge.py --report

# Tests
pytest tests/
```

## Architecture

```
customer_message
      |
      v
 intent classifier  (TF-IDF + LogisticRegression, scripts/train_intent_classifier.py)
      |
      v
 historical retriever (TF-IDF cosine similarity, scripts/retrieve_historical_replies.py)
      |  top-k evidence, each tagged response_type
      v
 reply generator (scripts/generate_support_reply.py)
      |  MOCK mode: template grounded only in top-1 retrieved evidence
      |  REAL mode: provider interface exists (Anthropic), unused -- no API key configured
      v
 AUTO-HANDLE / ESCALATE decision (scripts/decide_handling.py, 9 explicit rules)
      |
      v
 draft reply + decision + reason
```

Evaluation sits alongside, not inside, this pipeline:

```
golden_eval.csv (200 examples) --> evaluate_agent.py --> evaluation_results.json / evaluation_predictions.csv
                                                       --> llm_judge.py --> llm_judge_results.json / llm_judge_predictions.csv
                                                       --> human_judge.py --> human_judge_results.csv (0/30 scored -- see Limitations)
```

## Results

All numbers below are real, taken from `data/processed/evaluation_results.json`
and `data/processed/llm_judge_results.json`. **These are development/
evaluation results measured against a mostly-LLM-generated reference set,
not verified real-world accuracy** — see the warning above and Limitations
below before treating any number here as a real-world correctness claim.
Full breakdown, per-class metrics, confusion matrices, and 10 real failure
examples are in [reports/final_report.md](reports/final_report.md).

| Metric | System | Baseline |
|---|---|---|
| Intent accuracy | 0.470 | 0.410 (majority class) |
| Intent macro F1 | 0.303 | 0.065 |
| Intent weighted F1 | 0.444 | 0.238 |
| Handling accuracy | 0.410 | 0.370 (always ESCALATE) |
| AUTO-HANDLE precision / recall / F1 | 0.722 / 0.103 / 0.181 | 0.000 / 0.000 / 0.000 |
| Response-behavior accuracy | 0.335 | n/a |
| LLM-judge overall (1-5) | 4.33 | n/a |
| — relevance / groundedness / helpfulness / safety | 4.25 / 4.97 / **3.15** / 4.95 | n/a |

**Read this before quoting any single number** — see
[reports/final_report.md, Section 5](reports/final_report.md#5-what-is-misleading-about-my-headline-number)
for why "72% AUTO-HANDLE precision" and "4.33/5 quality" are both easy to
misread out of context.

## Failure Analysis

See [reports/final_report.md, Section 4](reports/final_report.md#4-top-5-failure-modes)
for the top 5 failure modes, each with a real example (tweet_id, exact
text, and likely cause) — not summarized here to avoid drifting out of
sync with the source data.

## What Is Misleading About My Headline Number?

See [reports/final_report.md, Section 5](reports/final_report.md#5-what-is-misleading-about-my-headline-number).

## What I Would Do With One More Week

See [reports/final_report.md, Section 6](reports/final_report.md#6-what-i-would-do-with-one-more-week).

## Limitations

- **The 200-example golden evaluation set is 10 human-labelled + 190
  LLM-generated** (bulk auto-approved without individual human review — see
  `docs/decision_log.md` #10-12). It is **not** a fully hand-labelled
  ground-truth set, and every metric above inherits that limitation.
- **Human-vs-LLM-judge agreement was never measured.** `scripts/human_judge.py`
  built a stratified 30-example sample and the tooling is complete and
  tested, but `data/processed/human_judge_results.csv` has **0/30 rows
  actually scored by a person**. There is no evidence the LLM judge agrees
  with human judgment.
- The intent classifier's own 435-example training set is LLM-assisted
  "silver" labels (Step 9), not hand-labelled either.
- The reply generator runs only in **mock/template mode** — no LLM API key
  is configured, so no generated text in this project came from a real
  language model; it is entirely retrieved-text reuse or safe refusal.
- Response-behavior accuracy (0.335) is capped by a real, disclosed
  vocabulary mismatch between the system's 5-category `response_type` and
  the golden set's 6-category `expected_response_behavior` — see
  `docs/decision_log.md` #14.
- Full list, with reasoning, in [reports/final_report.md, Section 7](reports/final_report.md#7-evaluation-limitations).

## Reproduction Instructions

1. Place the raw Kaggle CSV at `data/raw/twcs.csv` (gitignored — not part
   of this repo).
2. Run the commands in **Run Commands** above, in order. Every script
   prints its inputs/outputs and is deterministic (fixed seeds throughout;
   see `docs/decision_log.md` and `docs/project_notes.txt` for which seed
   is used where).
3. **`data/processed/*` is currently gitignored** (blanket rule from
   initial scaffolding — see `.gitignore`), so a fresh clone will not have
   `golden_eval.csv`, `evaluation_results.json`, `llm_judge_results.json`,
   etc. already present; they are regenerated by running the scripts above.
   This includes the golden evaluation set itself, which this project's
   real run history produced via `scripts/prepare_golden_set.py` +
   `scripts/bulk_approve_golden_labels.py` rather than a full manual
   labeling pass (see Limitations).
4. `scripts/human_judge.py` **must be run by an actual person** in an
   interactive terminal to produce real human-vs-LLM-judge agreement
   numbers — this was not done in this project's history.

## Repository Structure

```
hiver-sde-support-agent/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── src/
│   └── hiver_agent/
│       ├── __init__.py
│       └── config.py
├── data/
│   ├── raw/            (gitignored)
│   ├── processed/      (gitignored -- see Reproduction Instructions)
│   └── README.md
├── notebooks/
├── tests/
│   ├── __init__.py
│   └── test_generate_support_reply.py
├── scripts/             19 steps' worth of scripts -- see Run Commands
├── configs/             intent taxonomy + labeling guides
├── evaluations/
├── reports/
│   └── final_report.md
└── docs/
    ├── decision_log.md
    └── project_notes.txt
```

## Decision Log

See [docs/decision_log.md](docs/decision_log.md).
