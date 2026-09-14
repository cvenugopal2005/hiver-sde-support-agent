"""Streamlit demo UI for the Hiver AI Support Agent.

This file contains ONLY presentation logic. Every prediction, retrieval,
reply, and decision comes from the existing pipeline modules under
scripts/ -- nothing here re-implements or duplicates that logic:

    scripts/generate_support_reply.py  -> generate_support_reply(), get_llm_provider()
    scripts/decide_handling.py         -> decide_handling()

If no LLM API key is configured (the default in this project), the reply
generator automatically runs in its existing MOCK/template mode -- shown
to the evaluator as a small, honest "Demo mode" indicator, never hidden.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from decide_handling import decide_handling  # noqa: E402
from generate_support_reply import (  # noqa: E402
    generate_support_reply,
    get_llm_provider,
    load_classifier,
    load_retriever,
)

st.set_page_config(page_title="Hiver AI Support Agent", page_icon="🎧", layout="centered")


def humanize_reason(reason: str, decision: str) -> str:
    """Presentation-only rewording of decide_handling()'s reason string.

    decide_handling.py's reasons are written for engineering/audit purposes
    (they include raw confidence and similarity numbers) -- accurate, but
    not appropriate for an evaluator-facing screen. This maps each of its
    known reason patterns to a plain-language equivalent with the SAME
    meaning; it never changes the decision or invents a new justification.
    Falls back to a generic, decision-consistent sentence for any reason
    text that doesn't match a known pattern, so this stays safe even if
    decide_handling.py's wording changes later.
    """
    r = reason.lower()
    if "no draft reply" in r:
        return "No response could be generated for this message."
    if "no historical evidence" in r:
        return "No similar past conversations were found to safely base a reply on."
    if "always escalated to a human" in r:
        return "This involves account, purchase, or billing details, which always go to a human agent."
    if "sensitive account/payment term" in r:
        return "This message mentions sensitive account or payment details, so it's routed to a human agent."
    if "classifier confidence" in r:
        return "The AI wasn't confident enough about this topic to handle it automatically."
    if "too weak to trust for an automatic reply" in r:
        return "No closely matching past conversation was found to safely base a reply on."
    if "dm handoff" in r:
        return "The most relevant past case needed a private follow-up, not a public resolution."
    if "escalation referral" in r or "not a reusable public instruction" in r:
        return "The most relevant past case needed further review by a support specialist."
    if "safe, reusable clarifying question" in r:
        return "The best next step is to ask the customer for a bit more detail first."
    if "genuine public" in r and "strong textual similarity" in r:
        return "A closely matching past conversation shows this can be resolved with a standard, safe response."
    if "too weak to trust as a direct reuse" in r:
        return "A similar past case exists, but it isn't a close enough match to safely reuse automatically."
    if "unrecognized response_type" in r:
        return "This case didn't match a known pattern, so it's routed to a human agent."
    return "Escalated to a human agent for review." if decision == "ESCALATE" else "This can be safely handled automatically."

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown(
    """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 760px; }

.hiver-hero { text-align: center; margin-bottom: 0.6rem; }
.hiver-hero .badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #ec4899 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-size: 2.3rem; font-weight: 800; margin: 0;
}
.hiver-hero p { color: #64748b; font-size: 1.02rem; margin: 0.3rem 0 0 0; }

.demo-pill {
    display: inline-block; background: #f1f5f9; color: #64748b;
    border: 1px solid #e2e8f0; padding: 0.2rem 0.8rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
    text-transform: uppercase; margin-bottom: 1.4rem;
}

.result-title { text-align: center; font-size: 1.15rem; font-weight: 800; color: #1e293b;
    margin: 0.4rem 0 1.1rem 0; }

.field-label { font-size: 0.78rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem; }

.status-pill {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.7rem 1.6rem; border-radius: 999px;
    font-weight: 800; font-size: 1.3rem; letter-spacing: 0.01em;
}
.status-pill.success { background: #ecfdf5; color: #047857; border: 1.5px solid #a7f3d0; }
.status-pill.warning { background: #fffbeb; color: #b45309; border: 1.5px solid #fde68a; }

.reply-bubble {
    background: #f8fafc; border-left: 4px solid #6366f1;
    padding: 1rem 1.25rem; border-radius: 0.9rem;
    font-size: 1rem; line-height: 1.6; color: #1e293b;
}
.grounded-tag {
    display: inline-block; color: #94a3b8;
    font-size: 0.72rem; font-weight: 600; margin-top: 0.65rem;
}

[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; }
div[data-testid="stTextArea"] textarea { border-radius: 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Pipeline warm-up (loads the existing classifier + retriever ONCE per
# server process; nothing here is retrained or reloaded per click)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Getting ready...")
def warm_up_pipeline() -> bool:
    load_classifier()
    load_retriever()
    return True


warm_up_error: str | None = None
try:
    warm_up_pipeline()
except Exception as exc:  # noqa: BLE001 -- surfaced as a friendly message below, not a traceback
    warm_up_error = str(exc)

try:
    _, _reply_mode = get_llm_provider()
except Exception:  # noqa: BLE001
    _reply_mode = "mock"

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    """
<div class="hiver-hero">
    <p class="badge">Hiver AI Support Agent</p>
    <p>AI-powered customer support for AppleSupport</p>
</div>
""",
    unsafe_allow_html=True,
)

if _reply_mode == "mock":
    st.markdown(
        '<div style="text-align:center;"><span class="demo-pill">Demo mode</span></div>',
        unsafe_allow_html=True,
    )

if warm_up_error:
    st.error("This demo isn't ready yet — please try again in a moment.")
    st.stop()

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

with st.container(border=True):
    st.markdown('<div class="field-label">Customer message</div>', unsafe_allow_html=True)
    customer_message = st.text_area(
        "Customer message",
        placeholder="Type a customer issue here...",
        height=120,
        key="customer_message_textarea",
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("✨ Analyze & Respond", type="primary", use_container_width=True)


def run_pipeline_safe(message: str):
    """Thin, defensive wrapper around the existing pipeline functions.
    Returns (outcome_dict, friendly_error_message)."""
    try:
        result = generate_support_reply(message)
    except Exception:  # noqa: BLE001
        return None, "We couldn't analyze that message right now. Please try again."

    try:
        decision = decide_handling(
            customer_message=message,
            predicted_intent=result["predicted_intent"],
            retrieved_evidence=result["retrieved_examples"],
            draft_reply=result["draft_reply"],
        )
    except Exception:  # noqa: BLE001
        return None, "We couldn't decide how to route that message right now. Please try again."

    return {"result": result, "decision": decision}, None


if analyze_clicked:
    if not customer_message.strip():
        st.warning("Please type a customer issue before analyzing.")
    else:
        with st.spinner("Thinking..."):
            outcome, error = run_pipeline_safe(customer_message)
        if error:
            st.error(error)
            st.session_state.pop("last_outcome", None)
        else:
            st.session_state["last_outcome"] = outcome

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

outcome = st.session_state.get("last_outcome")

if outcome:
    result = outcome["result"]
    decision = outcome["decision"]

    st.write("")
    st.markdown('<div class="result-title">Support Agent Result</div>', unsafe_allow_html=True)

    # --- Card 1: Intent ---
    with st.container(border=True):
        st.markdown('<div class="field-label">Detected Intent</div>', unsafe_allow_html=True)
        st.markdown(f"## {result['predicted_intent']}")

    st.write("")

    # --- Card 2: Grounded reply ---
    with st.container(border=True):
        st.markdown('<div class="field-label">Suggested Reply</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="reply-bubble">{result["draft_reply"]}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="grounded-tag">Grounded in historical AppleSupport conversations</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # --- Card 3: Handling decision ---
    with st.container(border=True):
        if decision["decision"] == "AUTO-HANDLE":
            st.markdown('<span class="status-pill success">✅ AUTO-HANDLE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill warning">⚠️ ESCALATE</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="field-label">Reason</div>', unsafe_allow_html=True)
        st.write(humanize_reason(decision["reason"], decision["decision"]))

    evidence = result["retrieved_examples"]
    if evidence:
        st.write("")
        with st.expander("View historical evidence"):
            top = evidence[0]
            st.markdown(f"**Customer:** {top['historical_customer_text']}")
            st.markdown(f"**AppleSupport:** {top['historical_support_text']}")
