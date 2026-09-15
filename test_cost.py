"""Unit tests for cost arithmetic in cost.py."""

import pytest

import cost

RATES = {
    "inputTokens": 1.0,
    "outputTokens": 5.0,
    "cacheReadInputTokens": 0.1,
    "cacheWriteInputTokens": 1.25,
}


def test_sums_input_and_output():
    usage = {"inputTokens": 1_000_000, "outputTokens": 200_000}
    assert cost.usage_cost(usage, RATES) == pytest.approx(2.0)


def test_includes_cache_tokens_excluded_from_input_count():
    usage = {
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadInputTokens": 1_000_000,
        "cacheWriteInputTokens": 1_000_000,
    }
    assert cost.usage_cost(usage, RATES) == pytest.approx(1.35)
