"""Step 11 demo: run a handful of real queries through HistoricalReplyRetriever
and print the top results. This is a demonstration, NOT an evaluation --
retrieval quality has not been measured yet (no golden set, no metrics).

Usage:
    python scripts/demo_retrieval_queries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve_historical_replies import HistoricalReplyRetriever, load_or_build_corpus  # noqa: E402

# A few real-style queries covering different intents from the Step 4 taxonomy.
# Two are run both with and without the intent hint, to show its (soft) effect.
QUERIES = [
    {
        "text": "My iPhone battery is draining really fast since I updated to iOS 11",
        "intent": "Battery & Performance",
    },
    {
        "text": "My iPhone battery is draining really fast since I updated to iOS 11",
        "intent": None,
    },
    {
        "text": "My WiFi keeps disconnecting and reconnecting on my iPhone",
        "intent": "Connectivity",
    },
    {
        "text": "All my photos disappeared from my iPhone after the update",
        "intent": "Apple Music / iCloud Sync & Data Loss",
    },
    {
        "text": "I was charged twice for my Apple Music subscription, how do I get a refund",
        "intent": "Account, Purchases & Billing",
    },
    {
        "text": "My iPhone screen cracked, what's the warranty repair process",
        "intent": "Hardware, Repair & Warranty",
    },
]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    section("LOADING RETRIEVAL CORPUS")
    corpus = load_or_build_corpus()
    print(f"corpus rows: {len(corpus):,}")
    retriever = HistoricalReplyRetriever(corpus)

    for q in QUERIES:
        section(f"QUERY: {q['text']!r}  (intent hint: {q['intent']})")
        results = retriever.retrieve(q["text"], top_k=3, intent=q["intent"])
        for rank, r in enumerate(results, start=1):
            print(f"\n[{rank}] similarity={r['similarity_score']:.3f}  "
                  f"intent={r['intent']}  response_type={r['response_type']}  "
                  f"tweet_id={r['source_tweet_id']}")
            print(f"    HISTORICAL CUSTOMER: {r['historical_customer_text'][:200]}")
            print(f"    HISTORICAL SUPPORT:  {r['historical_support_text'][:200]}")

    section("REMINDER")
    print(
        "This is a demonstration of the retrieval mechanism only. Retrieval "
        "QUALITY (are these actually the most useful historical matches?) has "
        "not been evaluated -- that requires the golden evaluation set, which "
        "does not exist yet."
    )


if __name__ == "__main__":
    main()
