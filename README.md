# Hiver AI Support Agent

## 1. Project Overview

This project builds an AI customer-support agent for **AppleSupport**, using
the ["Customer Support on Twitter" (TWCS)](data/raw/twcs.csv) Kaggle dataset.
Given a new customer message, the agent:

1. **Classifies intent** into one of 8 AppleSupport-specific categories.
2. **Drafts a historically grounded reply**, built only from retrieved past
   AppleSupport conversations — never invented from scratch.
3. **Decides AUTO-HANDLE vs. ESCALATE**, with an explicit, auditable reason.

**Architecture:**

```
Customer Message
      →  Intent Classification
      →  Historical Retrieval
      →  Grounded Reply
      →  Handling Decision
```

## 2. Demo Video

[Watch the Demo Video](https://drive.google.com/file/d/1IoRSdXUPN2qOVFkhoxzXK8pkv3nZrpL5/view?usp=sharing)

The demo walks through several real customer messages, showing different
detected intents, grounded replies, and both AUTO-HANDLE and ESCALATE
decisions produced by the live application.

## 3. Problem Framing

**What "good" means for this agent, in the AppleSupport context:**

- Correctly identifying the customer's underlying issue (intent).
- Grounding every reply in real historical AppleSupport behavior, rather
  than generating unsupported or hallucinated claims.
- Escalating to a human whenever the evidence is weak, the classifier is
  unsure, or the topic is sensitive (account, payment, purchase) — instead
  of guessing.

**What was intentionally not built**, to keep scope realistic for a
take-home project:

- A full production support platform (queueing, agent assignment, live
  ticketing) — this is a decision-support pipeline, not a helpdesk system.
- Reconstruction of what happens after a private DM handoff. ~53% of
  AppleSupport's public replies hand the conversation off to DM, and this
  project deliberately never treats a DM handoff as a resolved issue.
- A vector database or embedding-search infrastructure — retrieval uses
  TF-IDF cosine similarity, which is sufficient at this dataset's scale and
  keeps the system easy to audit.
- Fully autonomous handling of account, purchase, or billing issues — these
  are always escalated to a human by design, regardless of confidence.

## 4. Dataset and Intent Taxonomy

The [TWCS dataset](data/raw/twcs.csv) contains public customer-support
Twitter conversations across many brands. **AppleSupport** was selected
based on conversation depth and resolution-pair volume, then validated by
manually reviewing real conversation samples. The 8 intents below were
derived from those real AppleSupport conversations, not assumed in advance
(see `configs/intent_taxonomy.md`):

1. Battery & Performance
2. Connectivity
3. Apple Music / iCloud Sync & Data Loss
4. Account, Purchases & Billing
5. Hardware, Repair & Warranty
6. Messaging & Communication
7. Software / App Bug Report
8. General / Pre-Purchase Inquiry

## 5. How to Run Locally

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Kaggle dataset at `data/raw/twcs.csv` (not committed — see
`.gitignore`).

### Run Commands

Run in order from a fresh clone; `data/processed/*` is gitignored (except a
few evaluation artifacts), so most of these regenerate local intermediate
files:

```bash
# Data prep and intent taxonomy
python scripts/prepare_intent_data.py
python scripts/prepare_training_sample.py
python scripts/refine_training_labels.py

# Train the intent classifier
python scripts/train_intent_classifier.py

# Retrieval, reply generation, decision logic
python scripts/retrieve_historical_replies.py
python scripts/generate_support_reply.py "My battery drains fast since the update"
python scripts/decide_handling.py "My battery drains fast since the update"

# Golden evaluation set + evaluation harness
python scripts/prepare_golden_set.py
python scripts/evaluate_agent.py
python scripts/llm_judge.py

# Streamlit demo UI
streamlit run app.py
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 6. Results

Real numbers from `data/processed/evaluation_results.json` and
`data/processed/llm_judge_results.json`, evaluated on a 200-example golden
set:

| Metric | System | Baseline |
|---|---|---|
| Intent accuracy | 0.470 | 0.410 (majority class) |
| Intent macro F1 | 0.303 | 0.065 |
| Handling accuracy | 0.410 | 0.370 (always ESCALATE) |
| AUTO-HANDLE precision / recall | 0.722 / 0.103 | 0.000 / 0.000 |
| LLM-judge overall quality (1-5) | 4.33 | n/a |

**Important:** the golden set is **10 human-labelled + 190 LLM-generated**
examples, not a fully hand-labelled ground truth, and human-vs-LLM-judge
agreement was never measured. Full results, failure analysis, and honest
limitations are in [reports/final_report.md](reports/final_report.md).

## 7. Repository Structure

```
hiver-sde-support-agent/
├── app.py                 Streamlit demo UI
├── scripts/                pipeline, training, retrieval, evaluation scripts
├── configs/                intent taxonomy + labeling guides
├── data/                   raw (gitignored) and processed data
├── tests/
├── reports/
│   └── final_report.md    full results, failure analysis, limitations
└── docs/
    └── decision_log.md    key design decisions and why
```

## 8. Decision Log

See [docs/decision_log.md](docs/decision_log.md) for the reasoning behind
key design choices made throughout the project.
