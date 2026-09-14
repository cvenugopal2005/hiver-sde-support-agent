"""Step 12: grounded reply generation.

Pipeline: customer_message -> intent classifier -> historical retrieval
(top 5) -> LLM prompt (customer message + predicted intent + retrieved
evidence) -> draft reply.

GROUNDING RULE (hard constraint): the draft reply must never invent
troubleshooting steps, policies, refunds, guarantees, or technical facts that
aren't present in the retrieved historical evidence. If the evidence is weak
(low similarity) or mostly DM handoffs / clarifying questions, the safe
output is a clarifying question or a DM/escalation pointer -- never a
fabricated fix. A DM handoff in the historical data is NEVER treated as
proof the underlying issue was resolved (same rule as Step 11).

LLM PROVIDER
No LLM API key is configured in this environment (checked in Step 9: no
.env, no ANTHROPIC_API_KEY/OPENAI_API_KEY, no `claude` CLI). This module
defines a small provider abstraction (LLMProvider) so a real provider can be
plugged in later via environment variables, and ships a MockLLMProvider that
builds a reply with a deterministic template from retrieved evidence only --
no model call happens. The active mode is always reported explicitly in the
returned dict ("mode": "mock" or "real:<provider>") -- mock output is never
presented as if it came from a real LLM.

Env vars (see .env.example):
    LLM_PROVIDER          e.g. "anthropic" (optional; if unset, or its key is
                          missing, falls back to mock mode automatically)
    ANTHROPIC_API_KEY     required if LLM_PROVIDER=anthropic

Usage:
    python scripts/generate_support_reply.py "My battery drains fast since the update"
"""

from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve_historical_replies import HistoricalReplyRetriever, load_or_build_corpus  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CLASSIFIER_PATH = DATA_DIR / "intent_classifier.joblib"

TOP_K = 5
# Below this top-1 similarity, the closest historical match is considered too
# dissimilar to ground a concrete instruction in -- fall back to a
# clarifying question instead of stretching a weak match into a "fix".
WEAK_EVIDENCE_SIMILARITY = 0.15


# --------------------------------------------------------------------------
# LLM provider abstraction
# --------------------------------------------------------------------------


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class AnthropicLLMProvider(LLMProvider):
    """Real provider. Requires the `anthropic` package and ANTHROPIC_API_KEY.
    Not exercised in this environment (no key configured) -- implemented so
    the pipeline is a one-env-var change away from real LLM mode."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5"):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but the 'anthropic' package is not "
                "installed. Add it to requirements.txt and pip install it "
                "to use real LLM mode."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


def get_llm_provider() -> tuple[LLMProvider | None, str]:
    """Chooses a provider from environment variables.

    Returns (provider, mode). provider is None in mock mode -- callers must
    then build the reply with build_mock_reply() instead of provider.generate().
    """
    provider_name = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if provider_name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            return AnthropicLLMProvider(api_key=api_key), "real:anthropic"
        print(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set -- "
            "falling back to mock mode."
        )
    elif provider_name:
        print(f"Unknown LLM_PROVIDER={provider_name!r} -- falling back to mock mode.")

    return None, "mock"


# --------------------------------------------------------------------------
# Prompt construction (identical regardless of mode, so the mock path is
# demonstrably built from the same evidence a real LLM prompt would contain)
# --------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are drafting a concise customer-support reply for AppleSupport on Twitter.

Ground your reply ONLY in the retrieved historical examples below. Do not invent:
- troubleshooting steps not shown in the evidence
- policies, refunds, or guarantees
- technical facts not supported by the evidence

A "DM handoff" response_type means the historical case was moved to DMs, NOT
that it was resolved -- never claim an issue was resolved based on a DM
handoff alone.

If the evidence is weak (low similarity) or mostly DM handoffs / clarifying
questions, the safest reply is a clarifying question or a DM/escalation
pointer, not an invented solution. Keep the reply short (2-4 sentences), in
the same supportive tone as the historical examples."""


def build_prompt(customer_message: str, predicted_intent: str, retrieved: list[dict]) -> str:
    lines = [
        SYSTEM_INSTRUCTIONS,
        "",
        f"NEW CUSTOMER MESSAGE: {customer_message}",
        f"PREDICTED INTENT: {predicted_intent}",
        "",
        "RETRIEVED HISTORICAL EXAMPLES (most similar first):",
    ]
    for i, r in enumerate(retrieved, start=1):
        lines.append(
            f"\n[{i}] similarity={r['similarity_score']:.3f}  response_type={r['response_type']}\n"
            f"    historical customer message: {r['historical_customer_text']}\n"
            f"    historical AppleSupport reply: {r['historical_support_text']}"
        )
    lines.append("\nDraft a concise reply to the NEW CUSTOMER MESSAGE above.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mock (template) reply generation -- deterministic, evidence-only
# --------------------------------------------------------------------------


def _strip_leading_mention(text: str) -> str:
    """Strips a single leading "@handle " token (the Twitter reply-recipient
    mention). Cosmetic only -- the historical wording itself is unchanged."""
    return re.sub(r"^@\S+\s*", "", text.strip())


def build_mock_reply(retrieved: list[dict]) -> str:
    """Deterministic, template-based reply built ONLY from retrieved
    evidence. This is NOT an LLM call -- it exists so the pipeline is
    testable without an API key. It never invents steps/policies: it either
    reuses the top retrieved instruction/confirmation text, echoes a
    retrieved clarifying question, or falls back to a generic
    clarifying/DM prompt when the evidence is weak or non-instructional.

    The grounding decision is based on retrieved[0] -- the single MOST
    similar historical match -- not a vote across all top_k results. An
    earlier version voted across the top 5, but a majority of DM handoffs
    among the 4 next-closest (less similar) matches would then override a
    genuinely useful, on-topic instruction sitting at rank 1, which produced
    the same generic "please DM us" reply for almost every query regardless
    of what evidence was actually available. The other retrieved examples
    are still returned/shown for context; they just don't drive the template.
    """
    if not retrieved:
        return (
            "Thanks for reaching out! Could you share a bit more detail about "
            "what's happening so we can help?"
        )

    top = retrieved[0]
    weak_evidence = top["similarity_score"] < WEAK_EVIDENCE_SIMILARITY

    if weak_evidence or top["response_type"] in ("DM handoff", "other/unclear"):
        return (
            "Thanks for reaching out! We'd like to take a closer look at this "
            "with you -- could you send us a DM with a bit more detail so we "
            "can help further?"
        )

    grounded_text = _strip_leading_mention(top["historical_support_text"])

    if top["response_type"] == "clarifying question":
        return (
            f"Thanks for letting us know. {grounded_text} "
            "(Similar cases needed this detail before we could help further.)"
        )

    return (
        f"Thanks for reaching out! In similar cases we've suggested: "
        f"{grounded_text} Let us know if that helps."
    )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

_classifier = None
_retriever = None


def load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _classifier


def load_retriever() -> HistoricalReplyRetriever:
    global _retriever
    if _retriever is None:
        corpus = load_or_build_corpus()
        _retriever = HistoricalReplyRetriever(corpus)
    return _retriever


def generate_support_reply(customer_message: str, top_k: int = TOP_K) -> dict:
    """Runs the full grounded-reply pipeline for one new customer message.

    NOTE: the intent classifier (Step 10) was trained on full
    conversation_context strings, but at inference time for a brand-new
    incoming message we only have the single customer message -- there is no
    conversation yet. This is a real, acknowledged domain mismatch (see
    report limitations), not something papered over here.
    """
    classifier = load_classifier()
    retriever = load_retriever()

    predicted_intent = classifier.predict([customer_message])[0]
    retrieved = retriever.retrieve(customer_message, top_k=top_k, intent=predicted_intent)

    prompt = build_prompt(customer_message, predicted_intent, retrieved)
    provider, mode = get_llm_provider()

    if provider is None:
        draft_reply = build_mock_reply(retrieved)
    else:
        draft_reply = provider.generate(prompt)

    return {
        "customer_message": customer_message,
        "predicted_intent": predicted_intent,
        "retrieved_examples": retrieved,
        "llm_prompt": prompt,
        "mode": mode,
        "draft_reply": draft_reply,
    }


def main() -> None:
    message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "My iPhone battery is draining really fast since I updated to iOS 11"
    )

    result = generate_support_reply(message)
    print(f"CUSTOMER MESSAGE: {result['customer_message']}")
    print(f"PREDICTED INTENT: {result['predicted_intent']}")
    print(f"MODE: {result['mode']}")
    print("\nTOP RETRIEVED EVIDENCE:")
    for i, r in enumerate(result["retrieved_examples"], start=1):
        print(
            f"  [{i}] sim={r['similarity_score']:.3f} type={r['response_type']} "
            f"-- {r['historical_support_text'][:120]}"
        )
    print(f"\nDRAFT REPLY:\n{result['draft_reply']}")


if __name__ == "__main__":
    main()
