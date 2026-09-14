"""Step 11: historical-response retrieval over the real AppleSupport corpus.

Given a new customer message, retrieve textually similar historical
AppleSupport conversations (customer message + the AppleSupport reply that
followed it) using TF-IDF + cosine similarity. No vector database, no
embedding model, no LLM.

IMPORTANT DATA LIMITATION (carried over from Steps 3A/3B): a large share of
AppleSupport's public replies are just a DM handoff, not a visible
resolution. Every retrieved result is tagged with a `response_type` so a
DM handoff is never presented as if it were a confirmed fix:
    "direct instruction/link" | "clarifying question" | "DM handoff" |
    "confirmation/status" | "other/unclear"
(reusing, then collapsing, the finer-grained classify_support_reply()
categories already used and validated in scripts/validate_applesupport.py).

Corpus: EVERY AppleSupport thread with >=1 customer message and >=1
AppleSupport reply (reusing thread reconstruction from
scripts/analyze_brands.py and the brand filter from
scripts/explore_applesupport.py) -- not just the small 450-example labeled
sample, so retrieval has real breadth. Intent is attached only where a
genuine label exists (the 435 non-UNCLEAR Step 9 silver labels); the other
~80,000+ corpus rows simply have intent=None, and retrieval works fine
without it, as required.

Output (gitignored, deterministic/reproducible -- built from the full
population, not a random sample, so there is no seed to fix):
    data/processed/retrieval_corpus.csv

Usage:
    python scripts/retrieve_historical_replies.py     # (re)builds the corpus and prints stats
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_brands import build_thread_roots, compute_thread_stats, load_data  # noqa: E402
from explore_applesupport import BRAND  # noqa: E402
from validate_applesupport import classify_support_reply  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CORPUS_CSV = DATA_DIR / "retrieval_corpus.csv"
SILVER_LABELS_CSV = DATA_DIR / "intent_training_sample_silver.csv"

RESPONSE_TYPE_MAP = {
    "link_instructions": "direct instruction/link",
    "direct_troubleshooting": "direct instruction/link",
    "clarifying_question": "clarifying question",
    "dm_handoff": "DM handoff",
    "confirmation_resolved": "confirmation/status",
    "status_update": "confirmation/status",
    "escalation_referral": "other/unclear",
    "unclear_no_observable_resolution": "other/unclear",
}

INTENT_BOOST = 0.05  # soft preference, not a hard filter -- see retrieve()


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_corpus() -> pd.DataFrame:
    """One row per AppleSupport thread with >=1 customer msg and >=1 reply:
    the first customer message, and the first AppleSupport reply that
    chronologically follows it (falling back to the thread's first reply if
    none strictly follows, which is rare)."""
    df = load_data()
    root, _dangling = build_thread_roots(df)
    stats = compute_thread_stats(df, root)

    qualifying_roots = stats.index[
        (stats["brand"] == BRAND) & (stats["n_customer"] > 0) & (stats["n_support"] > 0)
    ]
    mask = np.isin(root, qualifying_roots.to_numpy())
    subset = df.loc[mask].copy()
    subset["_root"] = root[mask]
    subset["_ts"] = pd.to_datetime(subset["created_at"], errors="coerce", format="mixed")

    rows = []
    for _root_idx, thread in subset.groupby("_root"):
        thread = thread.sort_values("_ts")
        customer_rows = thread[thread["inbound"] == "True"]
        support_rows = thread[thread["inbound"] == "False"]
        if len(customer_rows) == 0 or len(support_rows) == 0:
            continue

        first_customer = customer_rows.iloc[0]
        later_support = support_rows[support_rows["_ts"] > first_customer["_ts"]]
        response_row = later_support.iloc[0] if len(later_support) else support_rows.iloc[0]

        raw_type = classify_support_reply(response_row["text"])
        rows.append(
            {
                "tweet_id": first_customer["tweet_id"],
                "thread_root_tweet_id": thread.iloc[0]["tweet_id"],
                "customer_text": first_customer["text"].replace("\n", " "),
                "support_text": response_row["text"].replace("\n", " "),
                "response_type": RESPONSE_TYPE_MAP[raw_type],
            }
        )

    corpus = pd.DataFrame(rows)

    # Attach intent ONLY where a genuine Step 9 silver label exists (and is
    # not UNCLEAR, which is a labeling escape hatch, not a real intent).
    # Every other row keeps intent = None -- no fabricated labels.
    if SILVER_LABELS_CSV.exists():
        silver = pd.read_csv(SILVER_LABELS_CSV, dtype=str)
        silver = silver[silver["intent"] != "UNCLEAR"][["tweet_id", "intent"]]
        silver["tweet_id"] = silver["tweet_id"].astype(corpus["tweet_id"].dtype)
        corpus = corpus.merge(silver, on="tweet_id", how="left")
    else:
        corpus["intent"] = None

    return corpus


def load_or_build_corpus() -> pd.DataFrame:
    if CORPUS_CSV.exists():
        return pd.read_csv(CORPUS_CSV, dtype={"tweet_id": "int64", "thread_root_tweet_id": "int64"})
    corpus = build_corpus()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(CORPUS_CSV, index=False)
    return corpus


class HistoricalReplyRetriever:
    """TF-IDF + cosine-similarity retrieval over historical AppleSupport
    customer messages. No vector database, no embeddings, no LLM."""

    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.corpus["customer_text"].fillna(""))

    def retrieve(self, query_text: str, top_k: int = 5, intent: str | None = None) -> list[dict]:
        query_vec = self.vectorizer.transform([query_text])
        # TfidfVectorizer output is L2-normalized, so the dot product IS the
        # cosine similarity -- no separate cosine_similarity() call needed.
        sims = (self.matrix @ query_vec.T).toarray().ravel()

        adjusted = sims.copy()
        if intent:
            matches = (self.corpus["intent"] == intent).to_numpy()
            adjusted = adjusted + np.where(matches, INTENT_BOOST, 0.0)

        order = np.argsort(-adjusted)[:top_k]
        results = []
        for i in order:
            row = self.corpus.iloc[i]
            results.append(
                {
                    "historical_customer_text": row["customer_text"],
                    "historical_support_text": row["support_text"],
                    "similarity_score": float(sims[i]),
                    "intent": row["intent"] if pd.notna(row["intent"]) else None,
                    "response_type": row["response_type"],
                    "source_tweet_id": int(row["tweet_id"]),
                    "thread_root_tweet_id": int(row["thread_root_tweet_id"]),
                }
            )
        return results


def main() -> None:
    section("BUILDING / LOADING RETRIEVAL CORPUS (reused thread reconstruction)")
    corpus = load_or_build_corpus()
    print(f"corpus rows (AppleSupport threads with a customer msg + reply): {len(corpus):,}")
    print(f"rows with a genuine intent label (from Step 9 silver labels, excl. UNCLEAR): "
          f"{corpus['intent'].notna().sum():,}")
    print(f"saved to: {CORPUS_CSV} (gitignored)")

    section("RESPONSE_TYPE DISTRIBUTION (full corpus)")
    print(corpus["response_type"].value_counts().to_string())

    section("FITTING TF-IDF RETRIEVER")
    retriever = HistoricalReplyRetriever(corpus)
    print(f"vocabulary size: {len(retriever.vectorizer.vocabulary_):,}")
    print("Retriever ready. Run scripts/demo_retrieval_queries.py for example queries.")


if __name__ == "__main__":
    main()
