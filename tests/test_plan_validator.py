"""Tests for validation and normalization of parsed data plans."""

import pandas as pd
import pytest

from src.plan_validator import PlanValidationError, validate_plan


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Provide numeric, numeric-text, and categorical columns."""
    return pd.DataFrame({
        "age": [20, 30, 40],
        "salary": ["50000", "70000", "90000"],
        "department": ["Sales", "Tech", "Sales"],
    })


def _plan(action: str = "plot", **updates: object) -> dict[str, object]:
    """Create a valid-by-default histogram plan."""
    plan: dict[str, object] = {
        "action": action, "plot_type": "histogram", "x": "age", "y": None,
        "group_by": None, "aggregation": None, "filters": [], "top_n": None,
        "title": "", "explanation": "",
    }
    plan.update(updates)
    return plan


def test_missing_plot_column_is_repaired(dataframe: pd.DataFrame) -> None:
    """Keep plot requests executable when a column reference cannot be found."""
    result = validate_plan(_plan(x="height"), dataframe)
    assert result["x"] is None
    assert any("could not match" in warning for warning in result["warnings"])


def test_unique_fuzzy_column_match_is_corrected(dataframe: pd.DataFrame) -> None:
    """Correct a close typo and provide a user-visible notice."""
    result = validate_plan(_plan(x="salry"), dataframe)

    assert result["x"] == "salary"
    assert "Using 'salary'" in result["warnings"][0]


def test_wrong_plot_type_is_repaired_and_bad_filter_operator_is_rejected(
    dataframe: pd.DataFrame,
) -> None:
    """Repair an incompatible chart while rejecting unsafe filter operators."""
    plan = validate_plan(_plan(x="department"), dataframe)
    from src.plan_executor import execute_plan

    result = execute_plan(plan, dataframe)
    assert result.figure is not None
    assert result.warnings
    with pytest.raises(PlanValidationError, match="operator"):
        validate_plan(_plan(
            "filter",
            plot_type=None,
            filters=[{"column": "age", "op": "contains", "value": "3"}],
        ), dataframe)


def test_case_insensitive_columns_and_numeric_strings_are_supported(
    dataframe: pd.DataFrame,
) -> None:
    """Normalize capitalization and accept numeric values stored as strings."""
    result = validate_plan(_plan(x="SALARY"), dataframe)
    correlation = validate_plan(_plan(
        "correlate", plot_type=None, x="age", y="salary"
    ), dataframe)

    assert result["x"] == "salary"
    assert correlation["y"] == "salary"


def test_duplicate_case_insensitive_columns_are_repaired() -> None:
    """Keep a plot runnable when duplicate names prevent exact resolution."""
    dataframe = pd.DataFrame([[1, 2]], columns=["Value", "value"])
    result = validate_plan(_plan(x="VALUE"), dataframe)
    assert result["x"] is None
    assert result["warnings"]


def test_unknown_action_is_rejected_and_plot_type_is_repaired(
    dataframe: pd.DataFrame,
) -> None:
    """Reject unknown actions while repairing unknown chart names."""
    with pytest.raises(PlanValidationError, match="action"):
        validate_plan(_plan(action="python"), dataframe)
    result = validate_plan(_plan(plot_type="code"), dataframe)
    assert result["plot_type"] == "auto"
    assert result["warnings"]
