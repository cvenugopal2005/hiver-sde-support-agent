"""Step 13 demo: run the full pipeline (intent -> retrieval -> draft reply ->
AUTO-HANDLE/ESCALATE decision) on several real-style customer queries and
print the decision and reason for each.

This demonstrates the decision rules run end-to-end -- it is NOT an
evaluation of decision quality/accuracy (no golden set, no metrics; same
caveat as every prior demo script in this project).

Usage:
    python scripts/demo_handling_decisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decide_handling import decide_handling  # noqa: E402
from generate_support_reply import generate_support_reply  # noqa: E402

# Real-style queries chosen to exercise different branches of decide_handling:
# genuine reusable instruction, safe clarifying question, DM-handoff-only
# evidence, sensitive billing intent, sensitive keyword despite a different
# predicted intent, and weak/unusual evidence.
QUERIES = [
    "My iPhone battery is draining really fast since I updated to iOS 11",
    "My WiFi keeps disconnecting and reconnecting on my iPhone",
    "All my photos disappeared from my iPhone after the update",
    "I was charged twice for my Apple Music subscription, how do I get a refund",
    "My iPhone screen cracked, what's the warranty repair process",
    "Whenever I try typing the letter i it autocorrects to a weird symbol",
    "Can you reset my Apple ID password, I forgot it",
    "My HomePod mini won't pair with my new Apple TV 4K at all",
]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    for query in QUERIES:
        result = generate_support_reply(query)
        decision = decide_handling(
            customer_message=query,
            predicted_intent=result["predicted_intent"],
            retrieved_evidence=result["retrieved_examples"],
            draft_reply=result["draft_reply"],
        )
        top = result["retrieved_examples"][0] if result["retrieved_examples"] else None

        section(f"CUSTOMER MESSAGE: {query!r}")
        print(f"INTENT:   {result['predicted_intent']}")
        if top:
            print(f"TOP EVIDENCE: similarity={top['similarity_score']:.3f} response_type={top['response_type']}")
        print(f"DECISION: {decision['decision']}")
        print(f"REASON:   {decision['reason']}")
        print(f"DRAFT REPLY: {result['draft_reply']}")

    section("REMINDER")
    print(
        "This demonstrates the decision RULES running end-to-end on real "
        "queries. Decision QUALITY/ACCURACY has not been evaluated -- that "
        "requires the golden evaluation set and an evaluation harness, "
        "neither of which exist yet."
    )


if __name__ == "__main__":
    main()
