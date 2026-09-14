# Decision Log

A plain list of non-obvious decisions made during this project, and why.

1. **Project structure** — Used a standard `src/`-layout Python package
   (`src/hiver_agent/`) with separate top-level directories for data, notebooks,
   tests, scripts, configs, evaluations, reports, and docs. This keeps runtime
   code, experiments, evaluation artifacts, and deliverables (report, decision
   log) clearly separated, which matches the deliverables the assignment asks
   for (repo, golden eval set, evaluation harness, report, decision log) without
   introducing any extra infrastructure.

2. **Python version** — Targeting Python 3.11. It's a stable, widely-available
   version with good library support for the kind of data/LLM work this
   assignment requires, and doesn't require justification beyond "a recent,
   stable interpreter."

3. **Brand selection used conversation depth and resolution-pair volume, not
   raw tweet count** — ranking the 108 support accounts purely by tweet
   volume would favor brands that just tweet a lot, not brands with rich,
   resolvable conversations to learn from. AppleSupport was chosen after
   inspecting per-brand thread depth, resolution pairs, and actual sampled
   conversations (Step 2B), then validated on two independent 400-thread
   samples before finalizing anything (Steps 3A/3B).

4. **Thread reconstruction uses only `in_response_to_tweet_id`, never
   `response_tweet_id`** — the latter can list multiple children per row
   (comma-separated) and is fully redundant with the parent pointer for
   grouping into threads. Using only the parent pointer avoids a specific
   pitfall (assuming one `response_tweet_id` per row) and lets thread-root
   finding be a simple, vectorized "pointer jumping" pass instead of a
   per-row Python loop.

5. **The 8-intent taxonomy added "General / Pre-Purchase Inquiry," which
   was not in the assignment's original example list, and merged two
   originally-separate candidate clusters each into "Apple Music / iCloud
   Sync & Data Loss" and "Account, Purchases & Billing"** — manual reading
   of real messages (Step 3A/3B) showed pre-purchase questions behave
   completely differently for the auto-handle/escalate decision than any
   malfunction report, and that the merged pairs share the same downstream
   handling behavior (DM handoff for identity/payment; sync fixes for
   iCloud/Music), so keeping them separate wouldn't have changed anything
   the taxonomy needs to predict.

6. **The iOS 11.1 "I" autocorrect bug was folded into "Software / App Bug
   Report" rather than given its own top-level intent or excluded** — it's
   a huge (~13% of samples), extremely well-grounded pattern (a single
   canned, reusable workaround reply), but it's tied to one dated,
   time-boxed incident (Nov 2017), not a durable support category. Giving
   it its own class would optimize the taxonomy for a one-time anomaly in
   this specific dataset window; excluding it would throw away the
   strongest evidence of clean, reusable grounding anywhere in the data.

7. **No LLM API key is configured in this environment, and rather than
   blocking or fabricating labels, Claude was explicitly dispatched as
   independent subagents to act as the LLM classifier/judge** — used for
   the 450 silver training labels (Step 9), the 191 golden-set label
   proposals (Step 15), and the 200 LLM-judge quality scores (Step 17).
   Every use is disclosed in the code and docs as "Claude acting as judge,
   no API key configured," never presented as a live API call, and each
   provider interface (`generate_support_reply.py`, `llm_judge.py`) is
   built so a real provider is a one-env-var swap away.

8. **The mock reply generator grounds on the single most-similar (rank-1)
   retrieved match only, not a vote across the top 5** — an earlier version
   voted across 5 results, but less-similar DM-handoff matches at ranks 2-5
   kept outvoting a genuinely useful instruction sitting at rank 1,
   collapsing almost every generated reply into the same generic fallback
   regardless of what evidence was actually available (see Step 12; this
   is also failure mode #3 in the final report — even rank-1-only grounding
   still misses better evidence at rank 2/3).

9. **The AUTO-HANDLE/ESCALATE confidence threshold (0.25) was calibrated
   against the classifier's real held-out accuracy-vs-confidence curve, not
   guessed** — a first attempt at 0.40 was checked against the Step 10 test
   set and found to keep only 9/66 (14%) of predictions, escalating almost
   everything. The real curve showed accuracy rising from 0.64 (all
   predictions) to 0.89 among predictions with confidence >=0.25, so 0.25
   was used instead.

10. **The golden evaluation set (Step 14) was built to be provably disjoint
    from the training pool by recomputing and excluding the exact prior
    sampling pool, not by trusting a new random seed** — the 1000-thread
    Step 5 labeling pool (which the 450 training examples are a subset of)
    was recomputed deterministically with its original seed, and every one
    of those thread roots was excluded before drawing 200 new threads with
    a brand-new seed. Zero overlap was asserted in code, not just assumed.

11. **"Proposed" and "approved" golden-set labels were kept in physically
    separate columns/files, not a status flag on one file** — so it is
    structurally impossible for an LLM-generated proposal to become a
    "human" ground-truth label without an explicit y/n write. When later
    instructed to bulk-approve the remaining proposals without individual
    review (Step 15 addendum #2), the bulk script still detected and
    skipped a row that had, in the meantime, actually been approved
    individually by a human through the interactive review tool, rather
    than blindly overwriting it.

12. **The golden set's real composition (10 human-labelled + 190
    LLM-generated, bulk-approved without individual review) is stated
    explicitly in the code, the JSON output, `docs/project_notes.txt`, and
    this report — never rounded up to "hand-labelled"** — even though this
    was built under explicit instruction to move fast without full manual
    review. The same standard applies to human/LLM-judge agreement: it is
    reported as "not measured" (0/30 scored) rather than omitted or implied.

13. **The majority-intent baseline (Step 16) is computed from the TRAINING
    label distribution, not the golden set's own label distribution** — a
    baseline that "peeks" at the evaluation labels it's being compared
    against would be an unfair, inflated comparison.

14. **Response-behavior comparison (Step 16) uses an explicitly lossy 5-to-6
    category mapping rather than silently forcing a false equivalence** —
    the system's `other/unclear` response_type merges what the golden set
    separately calls `escalation` and `unclear`, so golden `escalation` rows
    can never be matched by construction. This is documented as a real cap
    on the response-behavior accuracy metric (0.335), not swept into "the
    model needs more training."

15. **Every interactive human-labeling/review/scoring tool built in this
    project (Steps 15 and 18) was smoke-tested exclusively against
    disposable temp files, never the real deliverable file** — so it is
    impossible for a test run to accidentally leave Claude-generated data
    inside an artifact meant to represent genuine human input or ground
    truth (e.g. `human_judge_results.csv` was verified to still have 0/30
    real scores after every test).
