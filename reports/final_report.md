# Final Report — AppleSupport AI Customer-Support Agent

*Hiver SDE Intern take-home assignment. All numbers in this report are taken directly from `data/processed/evaluation_results.json` and `data/processed/llm_judge_results.json` (both committed alongside this report). Nothing here is estimated or invented.*

## 1. Problem framing

Customer-support teams on Twitter (AppleSupport, in this project) handle a high volume of short, often ambiguous public messages, most of which get triaged into a DM rather than resolved publicly. The task was to build an AI agent for **one** brand from the Kaggle "Customer Support on Twitter" dataset that, given a new customer message, can: classify the customer's intent, retrieve relevant historical precedent, draft a grounded reply, and decide whether that reply is safe to send automatically or must go to a human — then evaluate all of it honestly, including where it fails.

**Brand chosen: AppleSupport** (Step 2B), selected using conversation depth and resolution-pair volume rather than raw tweet count, and validated with two independent 400-thread samples (Steps 3A/3B) before finalizing an 8-intent taxonomy (Step 4, `configs/intent_taxonomy.md`).

A defining constraint discovered early and carried through every later step: **~51-53% of AppleSupport's public replies are DM handoffs**, not visible resolutions (Steps 3A/3B, confirmed at full-population scale in Step 11: 52.8% of 80,639 threads). Every component below was built to never treat a DM handoff as evidence an issue was resolved.

## 2. What we built

| Component | File | Method |
|---|---|---|
| Intent classifier | `scripts/train_intent_classifier.py` | TF-IDF (1-2 grams) + Logistic Regression, trained on 435 LLM-assisted "silver" labels (`intent_training_sample_silver.csv`) |
| Historical retriever | `scripts/retrieve_historical_replies.py` | TF-IDF cosine similarity over all 80,639 qualifying AppleSupport threads; every result tagged with a `response_type` (direct instruction/link, clarifying question, DM handoff, confirmation/status, other/unclear) |
| Reply generator | `scripts/generate_support_reply.py` | Template-based **mock mode** (no LLM API key configured) — grounds only in the single most-similar retrieved example, never inventing facts; a real-provider interface exists but is unused |
| AUTO-HANDLE/ESCALATE decision | `scripts/decide_handling.py` | 9 explicit, ordered, non-LLM rules (sensitive intent, sensitive keyword, classifier confidence ≥0.25, retrieval similarity, response_type) |
| Golden evaluation set | `scripts/prepare_golden_set.py`, `scripts/review_golden_labels.py`, `scripts/bulk_approve_golden_labels.py` | 200 examples, provably disjoint from training data — see Section 7 for its real composition |
| Evaluation harness | `scripts/evaluate_agent.py` | Runs the full pipeline on all 200 golden examples, scores against two trivial baselines |
| LLM-as-judge | `scripts/llm_judge.py` | 4-dimension (relevance/groundedness/helpfulness/safety) quality judging of all 200 generated replies, via 5 parallel Claude-subagent judges (no API key configured) |
| Human-vs-judge agreement tool | `scripts/human_judge.py` | Stratified 30-example sample + agreement metrics — **built but not yet run by a human**, see Section 7 |

Every script is reproducible from `data/raw/twcs.csv` alone; see the README for exact commands and order.

## 3. Results vs baselines

**Intent classification** (200 golden examples, 8 classes + UNCLEAR):

| Metric | System | Majority-class baseline |
|---|---|---|
| Accuracy | **0.470** | 0.410 |
| Macro F1 | **0.303** | 0.065 |
| Weighted F1 | **0.444** | 0.238 |

The baseline always predicts `Software / App Bug Report` (the training set's majority class, 189/435 — computed from training labels, not the golden set, to avoid peeking). The system's real edge is macro F1 (0.303 vs 0.065): it actually attempts small classes instead of ignoring them.

**AUTO-HANDLE / ESCALATE decision:**

| Metric | System | Always-ESCALATE baseline |
|---|---|---|
| Accuracy | **0.410** | 0.370 |
| AUTO-HANDLE precision | 0.722 | 0.000 |
| AUTO-HANDLE recall | 0.103 | 0.000 |
| AUTO-HANDLE F1 | **0.181** | 0.000 |

**Response-behavior accuracy:** 0.335 (system only; no baseline was requested for this metric; see Section 7 for a real vocabulary-mismatch caveat).

**Reply quality (LLM-as-judge, real judgments via Claude subagents, n=200):**

| Dimension | Average (1-5) |
|---|---|
| Relevance | 4.25 |
| Groundedness | **4.97** |
| Helpfulness | **3.15** |
| Safety | 4.95 |
| **Overall** | **4.33** |

## 4. Top 5 failure modes

**1. Intent misclassification on small/specific classes, caused by a tiny, imbalanced, LLM-silver-labeled training set.**
Example (tweet_id `109054`): *"Why is the Messenger app so glitchy on the [AppleSupport] Watch?!"* — golden intent `Messaging & Communication`, predicted `Software / App Bug Report`. Likely cause: `Messaging & Communication` had only 9 of 435 training examples (the smallest class); TF-IDF has no semantic understanding of "Messenger app... Watch" and defaults toward the largest catch-all class. Per-class F1 for this intent is 0.25, tied for worst.

**2. AUTO-HANDLE severely under-triggered by an overly blunt, single global confidence threshold.**
Example (tweet_id `185544`, the well-documented iOS 11.1 "I" autocorrect bug): golden `AUTO-HANDLE`, predicted `ESCALATE` — reason: *"Classifier confidence in 'Software / App Bug Report' is only 0.24 (below 0.25)."* This is arguably the single best-grounded case in the entire dataset (a canned, verbatim, highly-reusable workaround link — see Step 4's taxonomy notes), escalated anyway because confidence landed one hundredth below a threshold that only looks at classifier confidence, never retrieval strength. System-wide AUTO-HANDLE recall is 0.103 — only 18/200 examples are ever predicted AUTO-HANDLE at all.

**3. The reply generator ignores stronger evidence sitting one rank below its top match.**
Example (tweet_id `556073`): customer asks about a specific WiFi/battery-icon glitch; the retrieved evidence contained a directly relevant answer, but the generator's top-1-only grounding rule (see Step 12 decision log) fell back to a generic "please DM us" because the *rank-1* match happened to be a DM handoff. LLM-judge helpfulness score: **1/5** — *"Ignores a specific, correct answer available in evidence; asks for DM unnecessarily."*

**4. Context-blindness: the system ignores facts already stated in the current message.**
Example (tweet_id `453966`): *"can you help me? I have sent my problem in DM"* — the generated reply asks the customer to send a DM again. Same pattern in tweet_id `1703838` (*"check your DM"*). Likely cause: the generator has no dialogue-state tracking; it only reasons over retrieval similarity, never over what the current message itself already asserts.

**5. Rare but real hallucinated-resolution safety miss.**
Example (tweet_id `916473`): *"Hey, what is happening?"* — the generator reused a historical reply reading *"We're glad to hear it fixed itself!"*, asserting an unverified resolution with no support in the current message. This scored the lowest of all 200 judged replies (overall 1.5/5; relevance 1, groundedness 2, helpfulness 1, safety 2). Likely cause: retrieval-based grounding checks textual similarity, not whether the retrieved reply's *implicit claim* ("it's fixed") actually transfers to the new case.

## 5. What is misleading about my headline number?

Three numbers in this report would sound better out of context than they should:

- **"72% AUTO-HANDLE precision."** This is trivially easy to achieve by being maximally conservative: the system only ever predicts AUTO-HANDLE for 18 of 200 examples (9%). Precision computed over 18 predictions is fragile, and the honest number is recall: **10.3%**. A headline that leads with precision alone would misrepresent a system that essentially never auto-handles anything.
- **"Beats baseline on intent accuracy/macro-F1."** True, but measured against a golden set that is **95% LLM-generated** (Section 7). "Beating baseline" here partly measures agreement with another LLM's reasoning about the same taxonomy, not verified real-world correctness — the two systems may share correlated blind spots rather than being independent checks on each other.
- **"4.33/5 average reply quality."** This sounds like a strong, uniform result, but it's carried entirely by groundedness (4.97) and safety (4.95) — both structurally inflated because the mock reply generator is *architecturally incapable* of inventing facts (Step 12); it can only reuse retrieved text or refuse. That's not evidence of sophisticated language generation, it's evidence the system rarely attempts anything ambitious enough to fail on those axes. The dimension that actually reflects customer value — **helpfulness, 3.15/5** — is hidden inside that average, and it drops to **2.77** for DM-handoff-type replies (103 of 200 examples) and **2.17** for unclear-type replies.

## 6. What I would do with one more week

1. **Get real human labels.** Run `scripts/review_golden_labels.py` for real (per-row, not the Step 15-addendum bulk shortcut) to replace the 190 LLM-generated golden labels, and actually run `scripts/human_judge.py` to collect the 30 human-vs-LLM-judge scores that were never collected here.
2. **Fix top-1-only grounding** in the reply generator (failure mode #3) — consider all top-k retrieved candidates and choose the most informative *safe* one, not just whichever ranks first.
3. **Add a same-message-context check** (failure mode #4) — a cheap regex/keyword check for "already sent/DM'd" before defaulting to a DM-ask template.
4. **Replace the single global confidence threshold with an evidence-aware gate** (failure mode #2) — combine classifier confidence *and* retrieval similarity, so well-grounded cases like the "I" bug aren't escalated on a borderline classifier number alone.
5. **Rebalance/expand training data** for the worst-performing classes (`Messaging & Communication` n=9, `Hardware, Repair & Warranty`, `Connectivity` — all at 0.25 F1), ideally with real human labels rather than more silver labels.
6. **Wire up a real LLM provider** (the interface already exists in both `generate_support_reply.py` and `llm_judge.py`) and re-run the judge to see whether genuine generation — not template reuse — closes the helpfulness gap without hurting groundedness/safety.

## 7. Evaluation limitations

- **The 200-example golden evaluation set is 10 human-labelled + 190 LLM-generated.** The 190 were produced as LLM proposals (Step 15) and then **bulk auto-approved without individual human review** (Step 15 addendum #2), on explicit instruction to finish the set quickly. **This is not a fully hand-labelled evaluation set and must not be represented as one.** Every metric in Sections 3-5 measures agreement with this mostly-LLM-labeled reference, not independently verified ground truth.
- **Human-vs-LLM-judge agreement was NOT measured.** `scripts/human_judge.py` (Step 18) built a stratified 30-example sample covering all 8 intents, all 5 response behaviors, and both handling outcomes, and `data/processed/human_judge_results.csv` exists — but with **0 of 30 rows scored**. No human ever ran the tool. There is currently **no evidence** that the LLM judge's scores (Section 3-4) agree with human judgment at all.
- The intent classifier and the golden set's LLM-generated labels were both produced using the same 8-intent taxonomy and similar LLM-based reasoning (Claude subagents, Steps 9 and 15) — so agreement between them is not fully independent evidence of correctness.
- `response_type`/response-behavior comparison uses a lossy 5-to-6 vocabulary mapping (Step 16): the system's `other/unclear` bucket merges what the golden set separately calls `escalation` and `unclear`, so golden `escalation` rows can structurally never be matched. Response-behavior accuracy (0.335) is capped by this alone, independent of actual reply quality.
- The reply generator ran only in **mock mode** throughout (no LLM API key configured) — all quality numbers describe a template-reuse system, not a real generative model.
- Intent classifier training data (435 examples) is itself LLM-assisted "silver" labels (Step 9), not hand-labelled either.

## 8. Decision summary

The single throughline across 19 steps: **never let an unverified or LLM-derived artifact quietly pass as ground truth.** Concretely — the brand and taxonomy were chosen from real sampled evidence, not assumption (Steps 2-4); the reply generator is structurally incapable of inventing unsupported claims (Step 12); the AUTO-HANDLE/ESCALATE gate is fully rule-based and auditable, with its one tunable threshold calibrated from real held-out data rather than guessed (Step 13); the golden set's LLM-generated majority is disclosed in the data, the code, and this report rather than hidden (Steps 15-16); and every "real" LLM output in this project (silver labels, golden-label proposals, judge scores) is explicitly tagged with its provenance and never silently presented as human work. See `docs/decision_log.md` for the full list of specific, non-obvious decisions and their reasoning.
