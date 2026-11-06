"""Tests for Ask Your Data UI helpers."""

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from app import _ask_examples, _plan_interpretation


def test_examples_are_distinct_and_use_real_columns() -> None:
    """Provide five useful requests for mixed and single-column datasets."""
    mixed = _ask_examples(pd.DataFrame({
        "salary": [10, 20],
        "team name": ["Sales", "Tech"],
    }))
    numeric_only = _ask_examples(pd.DataFrame({"average score": [1, 2]}))
    category_only = _ask_examples(pd.DataFrame({"team name": ["A", "B"]}))

    for examples in (mixed, numeric_only, category_only):
        assert len(examples) == 5
        assert len(set(examples)) == 5
    assert any("salary" in command for command in mixed)
    assert any("team name" in command for command in mixed)
    assert all("team name" in command or command == "count rows" for command in category_only)


def test_interpretation_describes_a_validated_filter() -> None:
    """Explain a validated plan before showing the filter result."""
    explanation = _plan_interpretation(
        {
            "action": "filter",
            "filters": [{"column": "age", "op": ">", "value": 30}],
        },
        "show rows where age > 30",
    )

    assert explanation == "I understood: show rows matching age > 30."
