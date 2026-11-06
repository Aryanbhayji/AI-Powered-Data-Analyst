"""Tests for privacy-safe summaries and AI fallback behavior."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.ai_explainer import (
    build_analysis_summary,
    explain_summary,
    generate_fallback_explanation,
)
from src.analysis import analyze_dataframe


def test_summary_contains_profile_but_not_raw_rows() -> None:
    """Include useful statistics while omitting cell-level dataset contents."""
    dataframe = pd.DataFrame({
        "sales": [10, 20, 500],
        "customer_note": ["private alpha", "private beta", "private gamma"],
    })
    summary = build_analysis_summary(analyze_dataframe(dataframe))

    assert "3 rows x 2 columns" in summary
    assert "sales" in summary
    assert "private alpha" not in summary


def test_missing_api_key_uses_offline_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explanations available without configured credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    summary = "Dataset shape: 3 rows x 2 columns.\nMissingness: none."

    result = explain_summary(summary, provider="openai")

    assert result.used_ai is False
    assert "No OPENAI_API_KEY" in result.notice
    assert "3 rows x 2 columns" in result.text


def test_provider_failure_uses_rule_based_fallback() -> None:
    """Fall back if the optional provider fails without making a network call."""
    summary = (
        "Dataset shape: 8 rows x 2 columns.\n"
        "Missingness: age (25.0%).\n"
        "Strongest correlations: age and score (r=0.90).\n"
        "IQR outliers: score: 1."
    )
    with patch("src.ai_explainer._get_model_response", side_effect=TimeoutError("timeout")):
        result = explain_summary(summary, provider="openai", api_key="test-key")

    assert result.used_ai is False
    assert "failed" in result.notice
    assert "age (25.0%)" in result.text
    assert "age and score" in result.text


def test_fallback_is_available_from_summary_alone() -> None:
    """The offline explanation consumes no dataframe or row-level data."""
    explanation = generate_fallback_explanation(
        "Dataset shape: 1 rows x 1 columns.\nMissingness: none."
    )

    assert "No missing values" in explanation
    assert "Next" in explanation
