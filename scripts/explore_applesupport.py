"""Step 3A: AppleSupport conversation sampling and issue-pattern discovery.

Reuses the thread-reconstruction logic in scripts/analyze_brands.py (does not
reimplement it) to build a deterministic sample of AppleSupport
conversations, then supports transparent manual inspection: sample
statistics, diverse printed conversations, and customer-message keyword
frequency. No brand is finalized here, no intents are defined, and no
model/LLM/agent code is built.

Outputs (gitignored, intermediate analysis artifacts only):
    data/processed/applesupport_sample.csv
    data/processed/applesupport_sample_customer_messages.txt

Usage:
    python scripts/explore_applesupport.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_brands import build_thread_roots, compute_thread_stats, load_data  # noqa: E402

BRAND = "AppleSupport"
SEED = 42
N_THREADS_SAMPLE = 400
N_EXAMPLES_TO_PRINT = 27
EXAMPLE_TRUNCATE = 220

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAMPLE_CSV = OUT_DIR / "applesupport_sample.csv"
CUSTOMER_MSGS_TXT = OUT_DIR / "applesupport_sample_customer_messages.txt"

STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "was", "were", "have", "has",
    "had", "with", "this", "that", "just", "not", "but", "can", "get", "got",
    "now", "still", "any", "all", "out", "when", "why", "how", "what", "who",
    "will", "would", "could", "should", "been", "being", "than", "then",
    "there", "their", "they", "them", "she", "him", "his", "her", "its",
    "from", "about", "into", "over", "even", "also", "some", "more", "most",
    "much", "many", "every", "here", "where", "which", "while", "after",
    "before", "because", "since", "does", "did", "doing", "again", "back",
    "new", "one", "two", "day", "days", "week", "weeks", "time", "please",
    "thanks", "thank", "hey", "hello", "yes", "yeah", "okay", "amp", "via",
    "applesupport", "apple",
}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_sample(
    df: pd.DataFrame,
    root: np.ndarray,
    stats: pd.DataFrame,
    seed: int = SEED,
    n_threads: int = N_THREADS_SAMPLE,
) -> pd.DataFrame:
    """Deterministic sample of AppleSupport threads with >=1 customer msg
    and >=1 AppleSupport msg (i.e. genuinely resolvable exchanges).

    `seed` and `n_threads` are parameterized so a second, independent
    sample can be drawn (Step 3B) using this exact same method rather than
    a reimplemented one.
    """
    brand_threads = stats[
        (stats["brand"] == BRAND) & (stats["n_customer"] > 0) & (stats["n_support"] > 0)
    ]
    rng = np.random.default_rng(seed)
    root_ids = brand_threads.index.to_numpy()
    n = min(n_threads, len(root_ids))
    chosen = rng.choice(root_ids, size=n, replace=False)

    mask = np.isin(root, chosen)
    sample_df = df.loc[mask].copy()
    sample_df["_root"] = root[mask]
    return sample_df


def save_sample(sample_df: pd.DataFrame, path: Path = SAMPLE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(path, index=False)


def dump_customer_messages(sample_df: pd.DataFrame, path: Path = CUSTOMER_MSGS_TXT) -> None:
    cust = sample_df[sample_df["inbound"] == "True"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in cust["text"]:
            f.write(t.replace("\n", " ") + "\n")


def sample_stats(sample_df: pd.DataFrame) -> dict:
    sizes = sample_df.groupby("_root").size()
    n_customer = int((sample_df["inbound"] == "True").sum())
    n_support = int((sample_df["inbound"] == "False").sum())
    return {
        "n_threads_sampled": int(sizes.shape[0]),
        "mean_thread_len": float(sizes.mean()),
        "median_thread_len": float(sizes.median()),
        "pct_exactly_2_turns": float((sizes == 2).mean() * 100),
        "pct_3plus_turns": float((sizes >= 3).mean() * 100),
        "pct_4plus_turns": float((sizes >= 4).mean() * 100),
        "n_customer_messages": n_customer,
        "n_applesupport_messages": n_support,
    }


VS16 = "️"  # variation selector-16; appears in the "I️" iOS 11 autocorrect glitch
I_BUG_CANNED_REPLY_FRAGMENT = "work around the issue until"


def is_i_autocorrect_bug_message(text: str) -> bool:
    """Heuristic flag for the iOS 11.1 'I'-autocorrect-glitch cluster found in
    Step 3A: the literal variation-selector glyph, or an explicit mention of
    the resulting question-mark-box symptom, or 'autocorrect' + 'letter'."""
    lower = text.lower()
    return (
        VS16 in text
        or "question mark" in lower
        or ("autocorrect" in lower and ("letter" in lower or '"i"' in lower or "'i'" in lower))
    )


def clean_customer_text(t: str) -> str:
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return t


def keyword_frequency(sample_df: pd.DataFrame, top_n: int = 40) -> list[tuple[str, int]]:
    cust = sample_df[sample_df["inbound"] == "True"]
    counter: Counter = Counter()
    for t in cust["text"]:
        cleaned = clean_customer_text(t)
        words = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 2]
        counter.update(words)
    return counter.most_common(top_n)


def print_diverse_conversations(sample_df: pd.DataFrame, n: int) -> None:
    """Stratified pick across short/medium/long threads, not consecutive rows."""
    sizes = sample_df.groupby("_root").size()
    buckets: dict = {2: [], 3: [], "4+": []}
    for r, s in sizes.items():
        if s == 2:
            buckets[2].append(r)
        elif s == 3:
            buckets[3].append(r)
        else:
            buckets["4+"].append(r)

    rng = np.random.default_rng(SEED + 1)
    per_bucket = n // 3
    chosen: list = []
    for key in (2, 3, "4+"):
        arr = np.array(buckets[key])
        rng.shuffle(arr)
        chosen.extend(arr[:per_bucket].tolist())
    # top up if a bucket was smaller than per_bucket
    remaining = [r for key in (2, 3, "4+") for r in buckets[key] if r not in chosen]
    rng.shuffle(np.array(remaining)) if remaining else None
    i = 0
    while len(chosen) < n and i < len(remaining):
        chosen.append(remaining[i])
        i += 1

    for r in chosen:
        thread = sample_df[sample_df["_root"] == r].copy()
        thread["_ts"] = pd.to_datetime(thread["created_at"], errors="coerce")
        thread = thread.sort_values("_ts")
        print(f"--- conversation (root_row={r}, size={len(thread)}) ---")
        for _, row in thread.iterrows():
            label = "APPLESUPPORT" if row["inbound"] == "False" else "CUSTOMER"
            text = row["text"]
            if len(text) > EXAMPLE_TRUNCATE:
                text = text[:EXAMPLE_TRUNCATE] + "…"
            print(f"{label}:\n{text}\n")
        print()


def main() -> None:
    section("LOADING DATA + RECONSTRUCTING THREADS (reused from analyze_brands.py)")
    df = load_data()
    root, _dangling = build_thread_roots(df)
    stats = compute_thread_stats(df, root)

    sample_df = build_sample(df, root, stats)
    save_sample(sample_df)
    dump_customer_messages(sample_df)
    print(f"sample saved to: {SAMPLE_CSV} (gitignored)")
    print(f"customer messages dumped to: {CUSTOMER_MSGS_TXT} (gitignored)")

    section("TASK 3: SAMPLE STATISTICS (sample only, NOT full-population claims)")
    for k, v in sample_stats(sample_df).items():
        print(f"{k}: {v}")

    section("KEYWORD FREQUENCY — customer messages in sample (aid for manual clustering)")
    for w, c in keyword_frequency(sample_df):
        print(f"{w}: {c}")

    section(f"TASK 2: DIVERSE CONVERSATION EXAMPLES (~{N_EXAMPLES_TO_PRINT}, stratified by length)")
    print_diverse_conversations(sample_df, N_EXAMPLES_TO_PRINT)


if __name__ == "__main__":
    main()
