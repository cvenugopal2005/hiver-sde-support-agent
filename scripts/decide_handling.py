"""Step 13: AUTO-HANDLE vs ESCALATE decision.

Given a customer message, its predicted intent, the retrieved historical
evidence (Step 11), and the grounded draft reply (Step 12), decide whether
the draft reply is safe to send automatically or must go to a human agent.

No LLM is used for this decision -- it is a small set of simple, explicit,
deterministic rules over signals already produced by earlier steps:
intent, retrieval similarity, response_type of the closest historical
match, and classifier confidence. Every decision carries a concrete reason.

HARD RULE (carried over from Steps 11/12): a "DM handoff" response_type is
NEVER treated as evidence the underlying issue was resolved. The closest
historical match being a DM handoff always escalates.

Decision priority (first matching rule wins):
    1. No draft reply / no retrieved evidence           -> ESCALATE
    2. Sensitive intent (account/purchases/billing)     -> ESCALATE
    3. Sensitive keyword in the raw message              -> ESCALATE
    4. Low classifier confidence for predicted_intent    -> ESCALATE
    5. Weak top-1 retrieval similarity                   -> ESCALATE
    6. Top-1 response_type is DM handoff / other-unclear -> ESCALATE
    7. Top-1 response_type is a clarifying question      -> AUTO-HANDLE
    8. Top-1 is direct instruction/link or confirmation,
       AND similarity is strong                          -> AUTO-HANDLE
    9. Anything else                                     -> ESCALATE

Usage:
    python scripts/decide_handling.py "My WiFi keeps disconnecting"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_support_reply import generate_support_reply, load_classifier  # noqa: E402

# Same thresholds used in generate_support_reply.py for "too weak to ground in".
WEAK_SIMILARITY = 0.15
# Above this, a direct instruction/link or confirmation is trusted enough to
# send automatically.
STRONG_SIMILARITY = 0.30
# Below this classifier confidence in the predicted intent, don't trust the
# intent enough to auto-handle. Chosen empirically from the Step 10 test set
# (66 held-out silver-labeled examples), not an arbitrary round number: with
# 8 balanced classes the random baseline is ~0.125; accuracy among test
# predictions with confidence >= 0.25 is 0.89 vs 0.64 across all predictions
# (a 0.40 cutoff was tried first and rejected -- it kept only 14% of
# predictions, escalating almost everything regardless of evidence quality).
CONFIDENCE_THRESHOLD = 0.25

# "Sensitive" intents are always escalated regardless of evidence quality --
# billing/account issues can involve real money or account access, which is
# not something to auto-resolve from a text-similarity match.
SENSITIVE_INTENTS = {"Account, Purchases & Billing"}

# Independent safety net: escalate on these terms even if the intent
# classifier mispredicts (a real, observed failure mode from Step 12).
SENSITIVE_KEYWORDS = re.compile(
    r"password|passcode|apple ?id|credit card|payment|refund|charged|billing|"
    r"unauthorized|hacked|security code|verification code",
    re.IGNORECASE,
)

NEVER_AUTO_RESPONSE_TYPES = {"DM handoff", "other/unclear"}
INSTRUCTION_LIKE_RESPONSE_TYPES = {"direct instruction/link", "confirmation/status"}


def _get_intent_confidence(customer_message: str, predicted_intent: str) -> float | None:
    """Best-effort classifier confidence for the predicted intent. Returns
    None if the model doesn't support predict_proba or the label is
    unrecognized -- callers must treat None as "unknown", not "low"."""
    classifier = load_classifier()
    if not hasattr(classifier, "predict_proba"):
        return None
    classes = list(classifier.classes_)
    if predicted_intent not in classes:
        return None
    proba = classifier.predict_proba([customer_message])[0]
    return float(proba[classes.index(predicted_intent)])


def decide_handling(
    customer_message: str,
    predicted_intent: str,
    retrieved_evidence: list[dict],
    draft_reply: str,
) -> dict:
    """Returns {"decision": "AUTO-HANDLE" | "ESCALATE", "reason": str}."""

    if not draft_reply or not draft_reply.strip():
        return {"decision": "ESCALATE", "reason": "No draft reply was produced."}

    if not retrieved_evidence:
        return {
            "decision": "ESCALATE",
            "reason": "No historical evidence was retrieved -- nothing to ground an automatic reply in.",
        }

    if predicted_intent in SENSITIVE_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Predicted intent '{predicted_intent}' involves account/purchases/"
                "billing -- always escalated to a human regardless of evidence quality."
            ),
        }

    if SENSITIVE_KEYWORDS.search(customer_message):
        return {
            "decision": "ESCALATE",
            "reason": (
                "Customer message contains a sensitive account/payment term "
                "(e.g. password, refund, charged) -- escalated as a safety net "
                "independent of the predicted intent, since intent classification "
                "can be wrong."
            ),
        }

    confidence = _get_intent_confidence(customer_message, predicted_intent)
    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Classifier confidence in '{predicted_intent}' is only "
                f"{confidence:.2f} (below {CONFIDENCE_THRESHOLD}) -- too low to "
                "trust the predicted intent for an automatic reply."
            ),
        }

    top = retrieved_evidence[0]
    similarity = top["similarity_score"]
    response_type = top["response_type"]

    if similarity < WEAK_SIMILARITY:
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Closest historical match has similarity {similarity:.2f} "
                f"(below {WEAK_SIMILARITY}) -- too weak to trust for an automatic reply."
            ),
        }

    if response_type in NEVER_AUTO_RESPONSE_TYPES:
        if response_type == "DM handoff":
            detail = "a DM handoff, which is NOT proof the issue was actually resolved"
        else:
            detail = "unclear / an escalation referral, not a reusable public instruction"
        return {
            "decision": "ESCALATE",
            "reason": f"Closest historical match (similarity {similarity:.2f}) is {detail}.",
        }

    if response_type == "clarifying question":
        return {
            "decision": "AUTO-HANDLE",
            "reason": (
                f"Closest historical match (similarity {similarity:.2f}) is a safe, "
                "reusable clarifying question -- it asks for more detail rather than "
                "asserting any fact, so there is no risk of inventing information."
            ),
        }

    if response_type in INSTRUCTION_LIKE_RESPONSE_TYPES:
        if similarity >= STRONG_SIMILARITY:
            return {
                "decision": "AUTO-HANDLE",
                "reason": (
                    f"Closest historical match (similarity {similarity:.2f}) is a "
                    f"genuine public {response_type} with strong textual similarity "
                    "-- safe to reuse."
                ),
            }
        return {
            "decision": "ESCALATE",
            "reason": (
                f"Closest historical match is a {response_type}, but similarity is "
                f"only {similarity:.2f} (below {STRONG_SIMILARITY}) -- too weak to "
                "trust as a direct reuse."
            ),
        }

    return {
        "decision": "ESCALATE",
        "reason": f"Unrecognized response_type {response_type!r} for the closest match -- defaulting to escalation.",
    }


def main() -> None:
    message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "My iPhone battery is draining really fast since I updated to iOS 11"
    )
    result = generate_support_reply(message)
    decision = decide_handling(
        customer_message=message,
        predicted_intent=result["predicted_intent"],
        retrieved_evidence=result["retrieved_examples"],
        draft_reply=result["draft_reply"],
    )
    print(f"CUSTOMER MESSAGE: {message}")
    print(f"PREDICTED INTENT: {result['predicted_intent']}")
    print(f"DRAFT REPLY: {result['draft_reply']}")
    print(f"DECISION: {decision['decision']}")
    print(f"REASON: {decision['reason']}")


if __name__ == "__main__":
    main()
