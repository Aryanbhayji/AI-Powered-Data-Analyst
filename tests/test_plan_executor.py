"""Tests for trusted execution of validated analysis plans."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.plan_executor import (
    MAX_SCATTER_POINTS,
    MAX_TABLE_ROWS,
    execute_plan,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Provide mixed, numeric-text, and categorical sample columns."""
    return pd.DataFrame({
        "age": [20, 30, 40, 50],
        "salary": ["50000", "70000", "90000", "110000"],
        "department": ["Sales", "Tech", "Sales", "Tech"],
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "gender": ["F", "M", "F", "M"],
    })


def _plan(action: str, **updates: object) -> dict[str, object]:
    """Create a complete structured plan for a trusted action."""
    plan: dict[str, object] = {
        "action": action, "plot_type": None, "x": None, "y": None,
        "group_by": None, "aggregation": None, "filters": [], "top_n": None,
        "title": "Test plan", "explanation": "A test plan.",
    }
    plan.update(updates)
    return plan


def _close_result(result: object) -> None:
    """Close returned figures after assertions."""
    figure = getattr(result, "figure", None)
    if figure is not None:
        plt.close(figure)


@pytest.mark.parametrize(
    ("plot_type", "columns"),
    [
        ("histogram", {"x": "age"}),
        ("bar", {"x": "department", "y": "salary"}),
        ("line", {"x": "month", "y": "salary"}),
        ("scatter", {"x": "age", "y": "salary"}),
        ("box", {"y": "salary", "group_by": "department"}),
        ("pie", {"x": "department"}),
        ("heatmap", {}),
    ],
)
def test_each_plot_type_returns_a_figure(
    dataframe: pd.DataFrame, plot_type: str, columns: dict[str, str]
) -> None:
    """Render every supported chart type from a validated structured plan."""
    result = execute_plan(
        _plan("plot", plot_type=plot_type, **columns), dataframe
    )

    assert isinstance(result.figure, Figure)
    _close_result(result)


def test_aggregate_returns_a_table_and_optional_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Aggregate a numeric-string measure by category with or without a chart."""
    table_result = execute_plan(_plan(
        "aggregate", group_by="department", y="salary", aggregation="mean"
    ), dataframe)
    chart_result = execute_plan(_plan(
        "aggregate", plot_type="bar", x="department", group_by="department",
        y="salary", aggregation="mean",
    ), dataframe)

    assert table_result.table is not None
    assert table_result.table.set_index("department")["value"].to_dict() == {
        "Sales": 70000.0,
        "Tech": 90000.0,
    }
    assert isinstance(chart_result.figure, Figure)
    assert chart_result.table is not None
    _close_result(chart_result)


def test_filter_describe_correlate_and_count_actions(
    dataframe: pd.DataFrame,
) -> None:
    """Execute every non-plot action with useful text/table outputs."""
    filtered = execute_plan(_plan(
        "filter",
        filters=[{"column": "age", "op": ">", "value": 30}],
    ), dataframe)
    described = execute_plan(_plan("describe", x="age"), dataframe)
    correlated = execute_plan(_plan(
        "correlate", x="age", y="salary"
    ), dataframe)
    counted = execute_plan(_plan("count", group_by="department"), dataframe)
    total = execute_plan(_plan("count"), dataframe)

    assert filtered.table["age"].tolist() == [40, 50]
    assert "2 matching" in filtered.text
    assert described.table is not None
    assert "mean" in described.table.columns
    assert correlated.table is not None
    assert correlated.figure is not None
    assert "correlation" in correlated.text
    assert counted.table["count"].sum() == 4
    assert "4 row" in total.text


def test_filter_does_not_mutate_input_and_caps_returned_rows() -> None:
    """Preserve the source dataframe and cap large filter tables."""
    dataframe = pd.DataFrame({"value": range(MAX_TABLE_ROWS + 5)})
    before = dataframe.copy(deep=True)
    result = execute_plan(_plan(
        "filter",
        filters=[{"column": "value", "op": ">=", "value": 0}],
    ), dataframe)

    pd.testing.assert_frame_equal(dataframe, before)
    assert len(result.table) == MAX_TABLE_ROWS
    assert "Showing the first" in result.warnings[0]


def test_scatter_sampling_and_category_chart_caps(dataframe: pd.DataFrame) -> None:
    """Sample large scatter inputs and limit charts to top 20 plus Other."""
    large = pd.DataFrame({
        "x": np.arange(MAX_SCATTER_POINTS + 2),
        "y": np.arange(MAX_SCATTER_POINTS + 2) * 2,
    })
    scatter = execute_plan(_plan(
        "plot", plot_type="scatter", x="x", y="y"
    ), large)
    categories = pd.DataFrame({"category": [f"item-{i}" for i in range(25)]})
    pie = execute_plan(_plan("plot", plot_type="pie", x="category"), categories)

    assert "sample" in scatter.warnings[0]
    assert isinstance(scatter.figure, Figure)
    assert len(pie.figure.axes[0].patches) == 21
    _close_result(scatter)
    _close_result(pie)


def test_single_row_and_all_null_numeric_values_are_safe() -> None:
    """Return an informative figure for an all-null numeric column."""
    dataframe = pd.DataFrame({"age": [np.nan]})
    result = execute_plan(_plan("plot", plot_type="histogram", x="age"), dataframe)

    assert isinstance(result.figure, Figure)
    assert result.figure.axes[0].texts
    assert "No numeric values" in result.figure.axes[0].texts[0].get_text()
    _close_result(result)


def test_text_filter_supports_equality_only(dataframe: pd.DataFrame) -> None:
    """Reject ordered comparisons against text columns with a friendly error."""
    with pytest.raises(ValueError, match="Use == or !="):
        execute_plan(_plan(
            "filter",
            filters=[{"column": "department", "op": ">", "value": "Sales"}],
        ), dataframe)


def test_string_numeric_and_datetime_filters_work() -> None:
    """Compare numeric strings and date-like values using safe conversions."""
    dataframe = pd.DataFrame({
        "salary": ["10", "20", "30"],
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
    })
    numeric = execute_plan(_plan(
        "filter",
        filters=[{"column": "salary", "op": ">=", "value": "20"}],
    ), dataframe)
    date = execute_plan(_plan(
        "filter",
        filters=[{"column": "date", "op": ">", "value": "2025-01-15"}],
    ), dataframe)

    assert numeric.table["salary"].tolist() == ["20", "30"]
    assert date.table["date"].tolist() == ["2025-02-01", "2025-03-01"]
