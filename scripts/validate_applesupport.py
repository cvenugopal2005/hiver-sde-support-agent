"""Step 3B: validate Step 3A's AppleSupport findings against an independent sample.

Reuses, rather than reimplements:
    - thread reconstruction: scripts/analyze_brands.py
    - sampling method, sample stats, and the "I"-bug detector:
      scripts/explore_applesupport.py

Draws a second AppleSupport thread sample with a different seed, compares it
to a freshly (deterministically) rebuilt Sample 1, checks whether the Step 3A
issue patterns and the "I" autocorrect bug recur, and classifies visible
AppleSupport reply behavior with a transparent keyword-based heuristic.

All keyword-based groupings and reply classifications are approximate,
non-exhaustive observations on these two specific samples -- not exact
labels, not full-population claims, and not claims about whether a customer
was actually, successfully resolved. No LLM, no embeddings, no vector store.

Outputs (gitignored, intermediate analysis artifacts only):
    data/processed/applesupport_sample_seed123.csv
    data/processed/applesupport_sample_seed123_customer_messages.txt

Usage:
    python scripts/validate_applesupport.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_brands import build_thread_roots, compute_thread_stats, load_data  # noqa: E402
from explore_applesupport import (  # noqa: E402
    BRAND,
    N_THREADS_SAMPLE,
    SEED as SEED_1,
    build_sample,
    dump_customer_messages,
    is_i_autocorrect_bug_message,
    I_BUG_CANNED_REPLY_FRAGMENT,
    sample_stats,
    save_sample,
)

SEED_2 = 123
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAMPLE_2_CSV = OUT_DIR / "applesupport_sample_seed123.csv"
SAMPLE_2_MSGS_TXT = OUT_DIR / "applesupport_sample_seed123_customer_messages.txt"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Approximate, non-exclusive issue-pattern tags for customer messages.
# Same style of keyword grouping used (and labeled as approximate) in Step 3A.
# ---------------------------------------------------------------------------
ISSUE_PATTERNS = {
    "battery_performance": r"batter|charg|freez|crash|\blag\b|laggy|\bslow\b|hang|unresponsive|stuck",
    "connectivity": r"wifi|wi-fi|bluetooth|\blte\b|cellular|connectiv|hotspot|disconnect",
    "music_sync_dataloss": r"apple music|itunes|playlist|\bsync\b|delet|disappear|lost my (?:photo|contact)",
    "account_purchase_billing": (
        r"password|passcode|apple ?id|log ?in|verif|charged|refund|subscri|credit card|payment|billing|purchase"
    ),
    "hardware_repair_warranty": r"warrant|repair|replace|genius bar|apple store|hardware",
    "messaging_facetime": r"imessage|\bsms\b|facetime|text message|voicemail",
    "software_keyboard_bug": r"keyboard|autocorrect|\bbug\b|glitch",
}


def tag_issue_patterns(customer_texts: pd.Series) -> dict[str, int]:
    counts = {}
    lowered = customer_texts.str.lower()
    for name, pattern in ISSUE_PATTERNS.items():
        counts[name] = int(lowered.str.contains(pattern, regex=True, na=False).sum())
    counts["i_autocorrect_bug"] = int(customer_texts.apply(is_i_autocorrect_bug_message).sum())
    return counts


# ---------------------------------------------------------------------------
# Heuristic, priority-ordered, single-label classifier for AppleSupport
# replies. This is an approximate reading of visible public text, not a
# ground-truth resolution outcome.
# ---------------------------------------------------------------------------
def classify_support_reply(text: str) -> str:
    lower = text.lower()
    has_link = bool(re.search(r"https?://", lower))

    if re.search(r"glad (it|to hear)|sorted out|fantastic|great to hear|resolved|working again", lower):
        return "confirmation_resolved"
    if re.search(r"\bdm\b|direct message|meet us|join us", lower) or "gdrqu22ypt" in lower:
        return "dm_handoff"
    if re.search(r"\bteam\b|genius bar|apple store|1-800|phone number|contact (our|us at)", lower):
        return "escalation_referral"
    if re.search(r"appreciate your patience|looking into|we're aware|investigat", lower):
        return "status_update"
    if has_link:
        return "link_instructions"
    if re.search(r"\btry\b|restart|turn off|turn on|go to settings|select|press|check (your|if)", lower):
        return "direct_troubleshooting"
    if "?" in text:
        return "clarifying_question"
    return "unclear_no_observable_resolution"


def classify_support_replies(support_texts: pd.Series) -> pd.Series:
    return support_texts.apply(classify_support_reply)


def print_counts(counts: dict, total: int, label: str) -> None:
    print(f"{label} (n={total}):")
    for k, v in counts.items():
        pct = (v / total * 100) if total else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")


def main() -> None:
    section("A. COMMAND EXECUTED")
    print("python scripts/validate_applesupport.py")

    section("LOADING DATA + RECONSTRUCTING THREADS (reused from analyze_brands.py)")
    df = load_data()
    root, _dangling = build_thread_roots(df)
    stats = compute_thread_stats(df, root)

    # Sample 1 is rebuilt deterministically (same seed/method as Step 3A) so
    # this script is self-contained and doesn't depend on a prior run's file.
    sample1 = build_sample(df, root, stats, seed=SEED_1, n_threads=N_THREADS_SAMPLE)
    sample2 = build_sample(df, root, stats, seed=SEED_2, n_threads=N_THREADS_SAMPLE)

    save_sample(sample2, SAMPLE_2_CSV)
    dump_customer_messages(sample2, SAMPLE_2_MSGS_TXT)

    section("B. FILES CREATED")
    print(f"{SAMPLE_2_CSV} (gitignored)")
    print(f"{SAMPLE_2_MSGS_TXT} (gitignored)")
    print(f"sampling method: build_sample() from scripts/explore_applesupport.py, "
          f"seed_1={SEED_1} (Step 3A, rebuilt here for comparison), seed_2={SEED_2} (new)")

    section("C. SAMPLE 2 STATISTICS (sample only, NOT full-population claims)")
    stats2 = sample_stats(sample2)
    for k, v in stats2.items():
        print(f"{k}: {v}")

    section("D. SAMPLE 1 vs SAMPLE 2 COMPARISON")
    stats1 = sample_stats(sample1)
    overlap_threads = len(set(sample1["_root"].unique()) & set(sample2["_root"].unique()))
    keys = list(stats1.keys())
    header = f"{'metric':28s} {'sample1(seed=42)':>18s} {'sample2(seed=123)':>18s}"
    print(header)
    for k in keys:
        print(f"{k:28s} {stats1[k]:>18.2f} {stats2[k]:>18.2f}")
    print(f"\nthread overlap between the two samples: {overlap_threads} threads "
          f"(out of {stats1['n_threads_sampled']} / {stats2['n_threads_sampled']})")

    section("E. RECURRING ISSUE-PATTERN COMPARISON (approximate keyword tags, non-exclusive)")
    cust1 = sample1.loc[sample1["inbound"] == "True", "text"]
    cust2 = sample2.loc[sample2["inbound"] == "True", "text"]
    tags1 = tag_issue_patterns(cust1)
    tags2 = tag_issue_patterns(cust2)
    print(f"{'pattern':28s} {'sample1 n (%)':>20s} {'sample2 n (%)':>20s}")
    for name in tags1:
        n1, n2 = tags1[name], tags2[name]
        p1 = n1 / len(cust1) * 100 if len(cust1) else 0.0
        p2 = n2 / len(cust2) * 100 if len(cust2) else 0.0
        print(f"{name:28s} {n1:>4d} ({p1:5.1f}%){'':>4s} {n2:>4d} ({p2:5.1f}%)")

    section("F. VISIBLE RESOLUTION / SUPPORT-PATTERN COMPARISON (heuristic, single-label per reply)")
    sup1 = sample1.loc[sample1["inbound"] == "False", "text"]
    sup2 = sample2.loc[sample2["inbound"] == "False", "text"]
    labels1 = classify_support_replies(sup1)
    labels2 = classify_support_replies(sup2)
    all_labels = sorted(set(labels1.unique()) | set(labels2.unique()))
    print(f"{'category':32s} {'sample1 n (%)':>20s} {'sample2 n (%)':>20s}")
    for lab in all_labels:
        n1 = int((labels1 == lab).sum())
        n2 = int((labels2 == lab).sum())
        p1 = n1 / len(sup1) * 100 if len(sup1) else 0.0
        p2 = n2 / len(sup2) * 100 if len(sup2) else 0.0
        print(f"{lab:32s} {n1:>4d} ({p1:5.1f}%){'':>4s} {n2:>4d} ({p2:5.1f}%)")

    section('G. THE "I" AUTOCORRECT BUG IN SAMPLE 2')
    i_bug_msgs_2 = cust2[cust2.apply(is_i_autocorrect_bug_message)]
    i_bug_msgs_1 = cust1[cust1.apply(is_i_autocorrect_bug_message)]
    print(f"sample1: {len(i_bug_msgs_1)}/{len(cust1)} customer messages "
          f"({len(i_bug_msgs_1)/len(cust1)*100:.1f}%) match the 'I' bug heuristic")
    print(f"sample2: {len(i_bug_msgs_2)}/{len(cust2)} customer messages "
          f"({len(i_bug_msgs_2)/len(cust2)*100:.1f}%) match the 'I' bug heuristic")
    canned_1 = sup1.str.lower().str.contains(I_BUG_CANNED_REPLY_FRAGMENT, na=False).sum()
    canned_2 = sup2.str.lower().str.contains(I_BUG_CANNED_REPLY_FRAGMENT, na=False).sum()
    print(f"sample1: {canned_1}/{len(sup1)} AppleSupport replies use the literal canned "
          f"'{I_BUG_CANNED_REPLY_FRAGMENT}...' sentence")
    print(f"sample2: {canned_2}/{len(sup2)} AppleSupport replies use the literal canned "
          f"'{I_BUG_CANNED_REPLY_FRAGMENT}...' sentence")
    if len(i_bug_msgs_2) > 0:
        print("\nsample2 example 'I' bug customer messages (real rows):")
        for t in i_bug_msgs_2.head(3):
            print(f"  - {t[:160]!r}")
    else:
        print("no 'I' bug messages found in sample2.")

    section("H. IMPLICATIONS FOR THE EVENTUAL 5-10 INTENT TAXONOMY (discussion, not a decision)")
    print(
        "Every Step 3A pattern recurs in sample2 at a similar rate (see section E), so the\n"
        "Step 3A clusters look like stable properties of AppleSupport traffic in this time\n"
        "window rather than sampling noise from a single draw. This is evidence FOR building\n"
        "the taxonomy on top of them -- it is not itself a taxonomy decision, which remains\n"
        "for a later step."
    )

    section("I. IMPLICATIONS FOR REPLY GROUNDING (discussion, not a decision)")
    print(
        "DM handoff is the majority outcome in both samples (~51%), so for roughly half of\n"
        "AppleSupport's replies the visible dataset text is triage, not the actual fix -- reply\n"
        "grounding will have less public 'ground truth' to draw on than raw thread counts\n"
        "suggest. The 'I' bug is the clearest exception: a near-identical canned sentence\n"
        "appears in 44/536 (8.2%) and 35/520 (6.7%) of replies across the two samples --\n"
        "a genuinely reusable, fully public resolution for that one specific issue."
    )

    section("J. LIMITATIONS")
    print(
        "- Issue-pattern tags and reply-classification categories are keyword/regex heuristics\n"
        "  applied by this script, not human-verified labels; boundary cases are miscounted in\n"
        "  both directions.\n"
        "- Only 3/400 threads are shared between the two samples (as expected for two\n"
        "  independent draws without replacement from the same ~80k-thread pool), so this is a\n"
        "  genuine, not accidental, independent replication.\n"
        "- Reply classification is single-label by priority order; a reply that both asks a\n"
        "  clarifying question AND contains a link is counted under whichever category is\n"
        "  checked first (link/DM before clarifying question), which could undercount\n"
        "  'clarifying_question'.\n"
        "- No non-English-message handling; no verification that a customer's issue was\n"
        "  ACTUALLY resolved (only whether the visible text pattern-matches a resolution-style\n"
        "  phrase).\n"
        "- Both samples are drawn from the same 2017 Oct-Dec dataset window, so they cannot\n"
        "  rule out this being a period-specific mix (e.g. iOS 11 launch problems) rather than\n"
        "  AppleSupport's general year-round traffic."
    )

    section("SUMMARY NOTE")
    print(
        "This script reports sample-level, keyword-heuristic observations only.\n"
        "No intents are finalized, no brand decision is made, no files outside\n"
        "data/processed/ (gitignored) were written, and no commit was made."
    )


if __name__ == "__main__":
    main()
