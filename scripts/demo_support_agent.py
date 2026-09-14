"""Step 12 demo: run several real-style customer queries through the full
grounded reply-generation pipeline (intent classifier -> retrieval ->
draft reply) and print the result for each.

This demonstrates the pipeline runs end-to-end and is grounded in retrieved
evidence -- it is NOT an evaluation of reply quality (no golden set, no
metrics; same caveat as Steps 10 and 11).

Usage:
    python scripts/demo_support_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_support_reply import generate_support_reply  # noqa: E402

# Real-style queries covering different intents from the Step 4 taxonomy.
QUERIES = [
    "My iPhone battery is draining really fast since I updated to iOS 11",
    "My WiFi keeps disconnecting and reconnecting on my iPhone",
    "All my photos disappeared from my iPhone after the update",
    "I was charged twice for my Apple Music subscription, how do I get a refund",
    "My iPhone screen cracked, what's the warranty repair process",
    "Whenever I try typing the letter i it autocorrects to a weird symbol",
]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    for query in QUERIES:
        section(f"CUSTOMER MESSAGE: {query!r}")
        result = generate_support_reply(query)

        print(f"PREDICTED INTENT: {result['predicted_intent']}")
        print(f"MODE: {result['mode']}"
              + ("  (deterministic template, NOT a real LLM call)" if result["mode"] == "mock" else ""))

        print("\nTOP RETRIEVED HISTORICAL EXAMPLES:")
        for i, r in enumerate(result["retrieved_examples"], start=1):
            print(
                f"  [{i}] sim={r['similarity_score']:.3f} response_type={r['response_type']} "
                f"intent={r['intent']}"
            )
            print(f"      customer: {r['historical_customer_text'][:150]}")
            print(f"      support:  {r['historical_support_text'][:150]}")

        print(f"\nGENERATED DRAFT REPLY:\n  {result['draft_reply']}")

    section("REMINDER")
    print(
        "This demonstrates the pipeline mechanics only. Reply QUALITY has not "
        "been evaluated -- that requires the golden evaluation set and an "
        "evaluation harness, neither of which exist yet."
    )


if __name__ == "__main__":
    main()
