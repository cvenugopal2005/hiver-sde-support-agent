# Golden Evaluation Set — Human Labeling Guide

Practical guide for hand-labeling `data/processed/golden_eval.csv` (200
independently-sampled AppleSupport threads — see Step 14 in
`docs/project_notes.txt` for how they were chosen). This is the ground-truth
evaluation set: it must be labeled by a human, independently, without
looking at the classifier's predictions, the retriever's output, or the
Step 8/9 silver labels. Its whole purpose is to check those components
against something that never influenced them.

Read the full `conversation_context` for each row before labeling — later
messages often clarify what's actually going on.

## 1. `intent`

Use the exact same 8-intent taxonomy and rules as training-data labeling —
see `configs/labeling_guide.md` for the full inclusion/exclusion table,
primary-intent priority order, and the "I" autocorrect bug rule. Don't
re-derive different rules here; the point of a golden set is comparability.

Valid values: `Battery & Performance`, `Connectivity`,
`Apple Music / iCloud Sync & Data Loss`, `Account, Purchases & Billing`,
`Hardware, Repair & Warranty`, `Messaging & Communication`,
`Software / App Bug Report`, `General / Pre-Purchase Inquiry`, or `UNCLEAR`
if genuinely undecidable.

## 2. `expected_response_behavior`

What kind of reply the CUSTOMER's message actually calls for, judged from
the message itself (not from what AppleSupport happened to say in this raw
transcript, and not from what an automated system happened to retrieve).
Pick one:

| Value | Use when... |
|---|---|
| `direct_instruction` | The issue is specific and common enough that a known troubleshooting step, setting, or link would genuinely help without needing any personal/account detail first. |
| `clarifying_question` | Real help requires more detail first (device/OS version, when it started, what's already been tried) before any instruction would be safe or useful. |
| `dm_handoff` | Resolving this requires exchanging account-specific or personal information (serial number, order number, full name) that shouldn't go in a public reply. |
| `escalation` | This needs a human specialist, a store visit, a phone line, or a policy decision (e.g. repair authorization, refund approval) — more than a DM exchange, a real handoff to another process/team. |
| `confirmation/status` | The message is a follow-up on an already-known issue and just needs an acknowledgment or status update, not a new instruction. |
| `unclear` | You can't tell what response would actually help (e.g. the message itself is too vague, off-topic, or spam-like). |

Judge this independently of intent — e.g. two `Account, Purchases & Billing`
messages can have different `expected_response_behavior` values (one might
be answerable directly, another might need a DM for account lookup).

## 3. `expected_handling`: AUTO-HANDLE vs ESCALATE

Ask: *"Could a support system safely send a reply to this customer without
a human reviewing it first?"*

- **AUTO-HANDLE** — the issue is common, low-stakes, and has a well-known,
  safe, generic answer or a safe clarifying question that carries no risk of
  being wrong or requiring account access (typically pairs with
  `direct_instruction` or `clarifying_question` above).
- **ESCALATE** — anything involving money, refunds, account access,
  security, physical repair/warranty decisions, or where getting it wrong
  has real consequences; also anything you marked `dm_handoff`,
  `escalation`, or `unclear` above should almost always be `ESCALATE`.

**Important:** a `dm_handoff` in the raw AppleSupport reply is never, on its
own, evidence the issue was resolved. Don't let seeing "DM us" in the
transcript talk you into labeling `expected_handling` as anything other
than what the CUSTOMER's message on its own would call for.

When genuinely torn, prefer **ESCALATE** — this is a customer-support
safety net, and a false ESCALATE just costs a human's time, while a false
AUTO-HANDLE risks sending a wrong or inappropriate reply.

## 4. Ambiguous cases

- If a message plausibly fits two response-behavior categories, pick the
  more conservative one (the one closer to `escalation`/ESCALATE) and note
  the ambiguity in `labeling_notes`.
- If the thread contains multiple distinct issues, label based on the
  FIRST customer message's primary issue (same rule as intent labeling),
  and note the secondary issue in `labeling_notes` — this stays single-label.
- If the message is non-English and you can't confidently judge it, label
  `intent=UNCLEAR`, `expected_response_behavior=unclear`,
  `expected_handling=ESCALATE`, and note the language issue.
- Never leave a row blank. If truly stuck, use `UNCLEAR` / `unclear` /
  `ESCALATE` and explain why in `labeling_notes` — a small number of these
  is expected and fine; guessing to force a confident-looking label is not.

## What NOT to do while labeling

- Don't look at `data/processed/intent_classifier.joblib` predictions,
  `scripts/retrieve_historical_replies.py` output, or
  `scripts/decide_handling.py` output while labeling. This set is only
  useful as ground truth if it's independent of all of them.
- Don't treat the AppleSupport reply visible in `conversation_context` as
  the "correct answer" — you are labeling what SHOULD happen given the
  customer's message, not grading what AppleSupport actually did.
