"""Execute validated data plans with trusted pandas and plotting operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.plan_validator import validate_plan
from src.visualizations import create_command_plot

MAX_TABLE_ROWS = 1000
MAX_PLOT_CATEGORIES = 20
MAX_SCATTER_POINTS = 20_000
AGGREGATIONS = {"mean": "mean", "sum": "sum", "median": "median",
                "min": "min", "max": "max"}


@dataclass
class PlanExecutionResult:
    """A plot, table, or text result plus user-facing notices."""

    figure: Figure | None = None
    table: pd.DataFrame | None = None
    text: str | None = None
    warnings: list[str] = field(default_factory=list)
    charts: dict[str, Figure] = field(default_factory=dict)
    statistics: pd.DataFrame | None = None
    explanation: str | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: str | None = None


def _numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Convert a numeric or numeric-looking column on a copy."""
    return pd.to_numeric(dataframe[column].copy(), errors="coerce")


def _limit_categories(
    data: pd.DataFrame, category_column: str, value_column: str, top_n: int | None
) -> pd.DataFrame:
    """Keep up to 20 categories and combine the remainder as Other."""
    limit = min(top_n or MAX_PLOT_CATEGORIES, MAX_PLOT_CATEGORIES)
    if len(data) <= limit:
        return data
    sorted_data = data.sort_values(value_column, ascending=False).copy()
    visible = sorted_data.head(limit).reset_index(drop=True).copy()
    other_value = sorted_data.iloc[limit:][value_column].sum()
    visible.loc[len(visible)] = {category_column: "Other", value_column: other_value}
    return visible


def _placeholder_figure(title: str, message: str) -> Figure:
    """Create an explanatory figure for an empty or degenerate plot."""
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.text(0.5, 0.5, message, ha="center", va="center",
              transform=axis.transAxes)
    axis.set_title(title)
    return figure


def _plot_histogram(dataframe: pd.DataFrame, plan: dict[str, Any]) -> Figure:
    """Plot a numeric column as a histogram."""
    column = plan["x"]
    values = _numeric_series(dataframe, column).dropna()
    if values.empty:
        return _placeholder_figure(plan.get("title") or column, "No numeric values to plot")
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.histplot(values, kde=len(values) > 1, ax=axis, color="#4472C4")
    axis.set(title=plan.get("title") or f"Distribution of {column}", xlabel=column)
    figure.tight_layout()
    return figure


def _aggregate_data(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
    limit_result: bool = True,
) -> pd.DataFrame:
    """Aggregate a numeric measure or count rows for each group."""
    group_column = plan.get("group_by")
    aggregation = plan["aggregation"]
    working = dataframe.copy()
    measure = plan.get("y")
    if group_column is None:
        if aggregation == "count":
            value = working[measure].count() if measure else len(working)
        else:
            values = _numeric_series(working, measure)
            value = getattr(values, AGGREGATIONS[aggregation])()
        return pd.DataFrame({"value": [value]})
    if aggregation == "count":
        if measure:
            result = working.groupby(group_column, dropna=False)[measure].count()
        else:
            result = working.groupby(group_column, dropna=False).size()
    else:
        working[measure] = _numeric_series(working, measure)
        grouped = working.groupby(group_column, dropna=False)[measure]
        result = getattr(grouped, AGGREGATIONS[aggregation])()
    output = result.rename("value").reset_index()
    if plan.get("top_n"):
        output = output.sort_values("value", ascending=False).head(plan["top_n"])
    elif limit_result:
        output = output.sort_values("value", ascending=False)
    if limit_result:
        output = output.head(MAX_TABLE_ROWS)
    return output.reset_index(drop=True)


def _bound_grouped_plot_data(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
    data: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Limit grouped chart categories and compute an accurate Other aggregate."""
    limit = min(plan.get("top_n") or MAX_PLOT_CATEGORIES, MAX_PLOT_CATEGORIES)
    if len(data) <= limit:
        return data
    data = data.sort_values(value_column, ascending=False).copy()
    visible = data.head(limit).reset_index(drop=True).copy()
    if plan.get("top_n"):
        return visible

    hidden_categories = data.iloc[limit:][category_column]
    remaining = dataframe.loc[dataframe[category_column].isin(hidden_categories)]
    aggregation = plan.get("aggregation") or "mean"
    measure = plan.get("y")
    if aggregation == "count":
        other_value = remaining[measure].count() if measure else len(remaining)
    else:
        values = _numeric_series(remaining, measure)
        other_value = getattr(values, AGGREGATIONS[aggregation])()
    visible.loc[len(visible)] = {category_column: "Other", value_column: other_value}
    return visible


def _plot_bar(dataframe: pd.DataFrame, plan: dict[str, Any]) -> Figure:
    """Plot grouped results or category frequencies as a bounded bar chart."""
    x_column = plan.get("x") or plan.get("group_by")
    y_column = plan.get("y")
    categories_bounded = False
    if x_column is None:
        return _placeholder_figure("Bar chart", "Choose a category column")
    if plan.get("action") in {"aggregate", "count"} and plan.get("group_by"):
        data = _aggregate_data(dataframe, plan, limit_result=False)
        category_column, value_column = plan["group_by"], "value"
        data = _bound_grouped_plot_data(
            dataframe, plan, data, category_column, value_column
        )
        categories_bounded = True
    elif y_column:
        working = dataframe[[x_column, y_column]].copy()
        working[y_column] = _numeric_series(working, y_column)
        data = (
            working.groupby(x_column, dropna=False)[y_column]
            .mean().rename("value").reset_index()
        )
        category_column, value_column = x_column, "value"
        data = _bound_grouped_plot_data(
            dataframe,
            {**plan, "aggregation": "mean"},
            data,
            category_column,
            value_column,
        )
        categories_bounded = True
    else:
        data = (
            dataframe[x_column].dropna().astype(str).value_counts()
            .rename_axis(x_column).rename("value").reset_index()
        )
        category_column, value_column = x_column, "value"
    if data.empty:
        return _placeholder_figure(plan.get("title") or "Bar chart", "No values to plot")
    if not categories_bounded:
        data = _limit_categories(
            data, category_column, value_column, plan.get("top_n")
        )
    figure, axis = plt.subplots(figsize=(9, 4))
    sns.barplot(data=data, x=value_column, y=category_column, ax=axis, color="#70AD47")
    axis.set(title=plan.get("title") or f"{value_column} by {category_column}")
    figure.tight_layout()
    return figure


def _plot_line(dataframe: pd.DataFrame, plan: dict[str, Any]) -> Figure:
    """Plot a numeric measure against an ordered x-axis."""
    working = dataframe[[plan["x"], plan["y"]]].copy()
    working[plan["y"]] = _numeric_series(working, plan["y"])
    working = working.dropna()
    if working.empty:
        return _placeholder_figure(plan.get("title") or "Line chart", "No values to plot")
    figure, axis = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=working, x=plan["x"], y=plan["y"], marker="o", ax=axis)
    axis.set_title(plan.get("title") or f"{plan['y']} by {plan['x']}")
    figure.tight_layout()
    return figure


def _plot_scatter(dataframe: pd.DataFrame, plan: dict[str, Any]) -> tuple[Figure, list[str]]:
    """Plot two numeric columns and sample datasets larger than 20,000 rows."""
    working = dataframe[[plan["x"], plan["y"]]].copy()
    working[plan["x"]] = _numeric_series(working, plan["x"])
    working[plan["y"]] = _numeric_series(working, plan["y"])
    working = working.dropna()
    notices = []
    if len(working) > MAX_SCATTER_POINTS:
        working = working.sample(MAX_SCATTER_POINTS, random_state=0)
        notices.append(
            f"Showing a deterministic sample of {MAX_SCATTER_POINTS:,} points."
        )
    if working.empty:
        return _placeholder_figure(plan.get("title") or "Scatter plot", "No numeric values to plot"), notices
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=working, x=plan["x"], y=plan["y"], ax=axis, alpha=0.7)
    axis.set_title(plan.get("title") or f"{plan['y']} vs {plan['x']}")
    figure.tight_layout()
    return figure, notices


def _plot_box(dataframe: pd.DataFrame, plan: dict[str, Any]) -> Figure:
    """Plot a numeric distribution, optionally split by a category."""
    value_column = plan["y"]
    working = dataframe.copy()
    working[value_column] = _numeric_series(working, value_column)
    working = working.dropna(subset=[value_column])
    if working.empty:
        return _placeholder_figure(plan.get("title") or "Box plot", "No numeric values to plot")
    figure, axis = plt.subplots(figsize=(8, 4))
    group_column = plan.get("group_by")
    if group_column:
        top_categories = working[group_column].astype(str).value_counts().head(
            MAX_PLOT_CATEGORIES
        ).index
        working = working[working[group_column].astype(str).isin(top_categories)]
        grouped_values = [
            working.loc[working[group_column].astype(str) == category, value_column]
            .dropna().to_numpy()
            for category in top_categories
        ]
        axis.boxplot(
            grouped_values,
            tick_labels=list(top_categories),
            orientation="vertical",
            patch_artist=True,
            boxprops={"facecolor": "#70AD47"},
        )
    else:
        axis.boxplot(
            working[value_column].dropna(),
            orientation="horizontal",
            patch_artist=True,
            boxprops={"facecolor": "#ED7D31"},
        )
    axis.set(title=plan.get("title") or f"Distribution of {value_column}")
    figure.tight_layout()
    return figure


def _plot_pie(dataframe: pd.DataFrame, plan: dict[str, Any]) -> Figure:
    """Plot category frequencies with overflow grouped into Other."""
    column = plan["x"]
    counts = dataframe[column].dropna().astype(str).value_counts()
    if counts.empty:
        return _placeholder_figure(plan.get("title") or "Pie chart", "No categories to plot")
    data = counts.rename_axis(column).rename("value").reset_index()
    data = _limit_categories(data, column, "value", plan.get("top_n"))
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.pie(data["value"], labels=data[column], autopct="%1.1f%%", startangle=90)
    axis.set_title(plan.get("title") or f"Distribution of {column}")
    figure.tight_layout()
    return figure


def _plot_heatmap(dataframe: pd.DataFrame, plan: dict[str, Any]) -> Figure:
    """Plot correlations for numeric and numeric-string columns."""
    numeric_data: dict[str, pd.Series] = {}
    for column in dataframe.columns:
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        if converted.notna().any() or pd.api.types.is_numeric_dtype(dataframe[column]):
            numeric_data[str(column)] = converted
    numeric = pd.DataFrame(numeric_data)
    if numeric.shape[1] < 2:
        return _placeholder_figure("Correlation heatmap", "At least two numeric columns are needed")
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0,
                fmt=".2f", ax=axis)
    axis.set_title(plan.get("title") or "Numeric feature correlations")
    figure.tight_layout()
    return figure


def _execute_plot(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> PlanExecutionResult:
    """Dispatch a validated chart type to its trusted plotting function."""
    plot_type = plan["plot_type"]
    notices: list[str] = []
    plotters = {
        "histogram": _plot_histogram,
        "bar": _plot_bar,
        "line": _plot_line,
        "box": _plot_box,
        "pie": _plot_pie,
        "heatmap": _plot_heatmap,
    }
    if plot_type == "scatter":
        figure, notices = _plot_scatter(dataframe, plan)
    elif plot_type == "bar" and plan.get("action") in {"aggregate", "count"}:
        figure = _plot_bar(dataframe, plan)
    else:
        figure = plotters[plot_type](dataframe, plan)
    return PlanExecutionResult(figure=figure, warnings=notices)


def _execute_aggregate(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> PlanExecutionResult:
    """Aggregate rows by a category and return a bounded table or chart."""
    result = _aggregate_data(dataframe, plan)
    if plan.get("plot_type"):
        return PlanExecutionResult(
            figure=_plot_bar(dataframe, {**plan, "plot_type": "bar"}),
            table=result,
        )
    return PlanExecutionResult(table=result)


def _apply_filter(dataframe: pd.DataFrame, item: dict[str, Any]) -> pd.Series:
    """Build one boolean mask without evaluating user-provided expressions."""
    series = dataframe[item["column"]]
    operator = item["op"]
    value = item["value"]
    numeric_values = pd.to_numeric(series, errors="coerce")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    numeric = pd.api.types.is_numeric_dtype(series) or (
        numeric_values.notna().any() and numeric_value is not None
    )
    if numeric:
        left = numeric_values
        if numeric_value is None:
            raise ValueError(f"Filter value for '{item['column']}' must be numeric.")
        right = numeric_value
    elif operator not in {"==", "!="}:
        date_strings = series.dropna().astype(str)
        date_like = pd.api.types.is_datetime64_any_dtype(series) or (
            not date_strings.empty
            and date_strings.str.match(
                r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
            ).mean() >= 0.5
        )
        dates = (
            pd.to_datetime(series, errors="coerce", format="mixed")
            if date_like else pd.Series(pd.NaT, index=series.index)
        )
        date_value = (
            pd.to_datetime(value, errors="coerce", format="mixed")
            if date_like else pd.NaT
        )
        if date_like and dates.notna().any() and not pd.isna(date_value):
            left = dates
            right = date_value
        else:
            raise ValueError(
                f"Use == or != to filter text column '{item['column']}'."
            )
    else:
        left = series.astype("string")
        right = str(value)
    operations = {
        ">": lambda: left > right,
        "<": lambda: left < right,
        ">=": lambda: left >= right,
        "<=": lambda: left <= right,
        "==": lambda: left == right,
        "!=": lambda: left != right,
    }
    return operations[operator]().fillna(False)


def _execute_filter(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> PlanExecutionResult:
    """Filter a copy of the input and cap the number of returned rows."""
    filtered = dataframe.copy()
    for item in plan["filters"]:
        filtered = filtered.loc[_apply_filter(filtered, item)]
    total_rows = len(filtered)
    notices = []
    if total_rows > MAX_TABLE_ROWS:
        notices.append(
            f"Showing the first {MAX_TABLE_ROWS:,} of {total_rows:,} matching rows."
        )
    return PlanExecutionResult(
        table=filtered.head(MAX_TABLE_ROWS).copy(),
        text=f"Found {total_rows:,} matching row(s).",
        warnings=notices,
    )


def _execute_describe(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> PlanExecutionResult:
    """Return descriptive statistics for the requested column(s)."""
    target = dataframe[[plan["x"]]].copy() if plan.get("x") else dataframe.copy()
    if target.shape[1] == 0:
        return PlanExecutionResult(text="This dataset has no columns to describe.")
    description = target.describe(include="all").transpose().reset_index(names="column")
    return PlanExecutionResult(table=description.head(MAX_TABLE_ROWS))


def _execute_correlate(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> PlanExecutionResult:
    """Return a two-column Pearson correlation."""
    x_column, y_column = plan["x"], plan["y"]
    x_values = _numeric_series(dataframe, x_column)
    y_values = _numeric_series(dataframe, y_column)
    paired = pd.DataFrame({x_column: x_values, y_column: y_values}).dropna()
    if len(paired) < 2 or paired[x_column].nunique() < 2 or paired[y_column].nunique() < 2:
        return PlanExecutionResult(
            text="There are not enough distinct numeric values to calculate a correlation."
        )
    correlation = float(paired[x_column].corr(paired[y_column]))
    table = pd.DataFrame({
        "Column 1": [x_column],
        "Column 2": [y_column],
        "Correlation": [correlation],
        "Paired rows": [len(paired)],
    })
    return PlanExecutionResult(
        table=table,
        text=f"The correlation between {x_column} and {y_column} is {correlation:.3f}.",
    )


def _execute_count(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> PlanExecutionResult:
    """Count all rows or count rows per category."""
    group_by = plan.get("group_by")
    if group_by is None:
        return PlanExecutionResult(text=f"The dataset contains {len(dataframe):,} row(s).")
    counts = (
        dataframe.groupby(group_by, dropna=False).size()
        .rename("count").reset_index()
        .sort_values("count", ascending=False)
        .head(plan.get("top_n") or MAX_TABLE_ROWS)
        .reset_index(drop=True)
    )
    if plan.get("plot_type") == "bar":
        chart_plan = {
            **plan, "x": group_by, "group_by": group_by,
            "aggregation": "count", "action": "count",
        }
        return PlanExecutionResult(figure=_plot_bar(dataframe, chart_plan), table=counts)
    return PlanExecutionResult(table=counts.head(MAX_TABLE_ROWS))


def execute_plan(
    plan: dict[str, Any], dataframe: pd.DataFrame
) -> PlanExecutionResult:
    """Validate and execute a plan using only the application's trusted actions."""
    validated = validate_plan(plan, dataframe)
    working = dataframe.copy(deep=True)
    actions = {
        "aggregate": _execute_aggregate,
        "filter": _execute_filter,
        "describe": _execute_describe,
        "correlate": _execute_correlate,
        "count": _execute_count,
    }
    if validated["action"] in {
        "plot", "relationship", "correlate", "full_analysis"
    }:
        figure, plot_notice, rows_used = create_command_plot(
            working, validated, str(validated.get("title") or "")
        )
        result_stats = pd.DataFrame({
            "Statistic": ["Rows used", "Rows dropped"],
            "Value": [rows_used, max(0, len(working) - rows_used)],
        })
        explanation = (
            f"The plot uses {rows_used:,} row(s) with complete values for its "
            "selected columns. A plot can show association, but not prove causation."
        )
        if validated["action"] == "correlate":
            explanation = (
                f"The correlation plot uses {rows_used:,} complete row(s). "
                "Correlation shows association, not causation."
            )
        plot_description = str(validated.get("title") or validated.get("plot_type"))
        summary = (
            f"Requested analysis: {plot_description[:200]}\n"
            f"Rows used: {rows_used}\n"
            f"Rows dropped for missing plotted values: {max(0, len(working) - rows_used)}\n"
            "This summary contains no raw dataset rows. Explain what this chart "
            "shows in simple English and do not claim causation."
        )
        notices = [plot_notice] if plot_notice else []
        result = PlanExecutionResult(
            figure=figure,
            statistics=result_stats,
            table=result_stats,
            explanation=explanation,
            text=explanation,
            summary=summary,
            warnings=notices,
        )
        result.warnings = validated["warnings"] + result.warnings
        return result
    result = actions[validated["action"]](working, validated)
    result.warnings = validated["warnings"] + result.warnings
    return result
