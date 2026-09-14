"""Step 12 basic tests: pipeline components load and produce non-empty,
mock-mode output when no LLM API key is configured.

These are smoke tests for the pipeline mechanics, NOT an evaluation of reply
quality (that requires the golden evaluation set, which doesn't exist yet).

Usage:
    pytest tests/test_generate_support_reply.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_support_reply import (  # noqa: E402
    CLASSIFIER_PATH,
    generate_support_reply,
    get_llm_provider,
    load_classifier,
    load_retriever,
)


def test_classifier_loads():
    assert CLASSIFIER_PATH.exists(), f"expected trained classifier at {CLASSIFIER_PATH}"
    model = joblib.load(CLASSIFIER_PATH)
    assert hasattr(model, "predict")
    pred = model.predict(["My battery drains fast since the update"])
    assert len(pred) == 1
    assert isinstance(pred[0], str) and pred[0]


def test_retriever_returns_results():
    retriever = load_retriever()
    results = retriever.retrieve("My WiFi keeps disconnecting", top_k=5)
    assert len(results) == 5
    for r in results:
        assert r["historical_customer_text"]
        assert r["historical_support_text"]
        assert r["response_type"] in (
            "direct instruction/link",
            "clarifying question",
            "DM handoff",
            "confirmation/status",
            "other/unclear",
        )


def test_reply_generator_uses_mock_when_no_api_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider, mode = get_llm_provider()
    assert provider is None
    assert mode == "mock"

    result = generate_support_reply("My iPhone battery is draining fast")
    assert result["mode"] == "mock"


def test_reply_generator_falls_back_to_mock_if_provider_set_without_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider, mode = get_llm_provider()
    assert provider is None
    assert mode == "mock"


@pytest.mark.parametrize(
    "message",
    [
        "My iPhone battery is draining really fast since I updated to iOS 11",
        "My WiFi keeps disconnecting and reconnecting on my iPhone",
        "I was charged twice for my Apple Music subscription, how do I get a refund",
    ],
)
def test_no_empty_reply(message, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    result = generate_support_reply(message)
    assert result["draft_reply"].strip() != ""
    assert result["predicted_intent"]
    assert len(result["retrieved_examples"]) > 0


def test_load_classifier_is_cached():
    m1 = load_classifier()
    m2 = load_classifier()
    assert m1 is m2
