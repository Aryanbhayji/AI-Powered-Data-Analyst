"""Automatic exploratory charts and plain-English findings for uploaded data."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.analysis import calculate_missing_values
from src.type_inference import infer_column_types
from src.visualizations import (
    MAX_SCATTER_POINTS,
    category_bar,
    data_story_boxplot,
    data_story_heatmap,
    data_story_histogram,
    data_story_lineplot,
    data_story_scatter,
    missingness_bar_chart,
)

MAX_CHARTS = 6
MAX_HEATMAP_COLUMNS = 15
MAX_CATEGORY_LEVELS = 50


@dataclass
class VisualizationCard:
    """One recommended visualization with the reason it is useful."""

    title: str
    description: str
    figure: Figure


@dataclass
class VisualizationReport:
    """Dataset-level findings and a compact collection of recommended charts."""

    insights: list[str] = field(default_factory=list)
    charts: list[VisualizationCard] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def _sample_for_chart(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Bound row-heavy plot inputs while keeping a reproducible sample."""
    if len(dataframe) <= MAX_SCATTER_POINTS:
        return dataframe.copy()
    return dataframe.sample(MAX_SCATTER_POINTS, random_state=0).copy()


def _numeric_columns(
    dataframe: pd.DataFrame, types: dict[str, str]
) -> list[str]:
    """Choose numeric measures and omit ID-like and constant columns."""
    columns: list[str] = []
    for column, kind in types.items():
        if kind not in {"numeric_continuous", "numeric_discrete", "binary"}:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.nunique() > 1:
            columns.append(column)
    return columns


def _make_insights(dataframe: pd.DataFrame, types: dict[str, str]) -> list[str]:
    """Summarize data quality, useful distributions, and strong relationships."""
    insights: list[str] = []
    if dataframe.empty:
        return ["This file has column headers but no data rows to analyze."]

    missing = calculate_missing_values(dataframe)
    incomplete = missing[missing["Missing Count"] > 0]
    if incomplete.empty:
        insights.append("No missing cells were found.")
    else:
        highest = incomplete.iloc[0]
        insights.append(
            f"{highest['Column']} has the most missing data "
            f"({highest['Missing %']:.1f}%)."
        )

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        insights.append(f"{duplicate_count:,} duplicate row(s) were found.")
    else:
        insights.append("No duplicate rows were found.")

    for column, kind in types.items():
        if kind != "numeric_continuous":
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        median = float(values.median())
        skew = float(values.skew()) if len(values) > 2 else 0.0
        if abs(skew) >= 1:
            direction = "right" if skew > 0 else "left"
            insights.append(
                f"{column} is strongly skewed to the {direction}; "
                f"its median ({median:.3g}) is more representative than its "
                f"mean ({mean:.3g})."
            )
        else:
            insights.append(
                f"{column} averages {mean:.3g}, with a median of {median:.3g}."
            )
        if len(insights) >= 4:
            break

    numeric = _numeric_columns(dataframe, types)
    if len(numeric) > 1:
        correlations = dataframe[numeric].apply(
            pd.to_numeric, errors="coerce"
        ).corr()
        upper = correlations.where(
            np.triu(np.ones(correlations.shape), k=1).astype(bool)
        ).stack().dropna()
        if not upper.empty:
            pair = upper.abs().idxmax()
            value = float(correlations.loc[pair[0], pair[1]])
            if abs(value) >= 0.5:
                direction = "positive" if value > 0 else "negative"
                insights.append(
                    f"{pair[0]} and {pair[1]} have a strong {direction} "
                    f"association (correlation {value:.2f}); this does not "
                    "show that one causes the other."
                )
    return insights[:6]


def build_visualization_report(dataframe: pd.DataFrame) -> VisualizationReport:
    """Scan a dataframe and build up to six analyst-style visualizations."""
    report = VisualizationReport()
    if dataframe.shape[1] == 0:
        report.insights = ["This file has no columns to visualize."]
        return report

    types = infer_column_types(dataframe)
    report.insights = _make_insights(dataframe, types)
    if dataframe.empty:
        return report

    plot_data = _sample_for_chart(dataframe)
    if len(plot_data) < len(dataframe):
        report.notices.append(
            f"Distribution, trend, grouped, and scatter charts use a "
            f"reproducible sample of {len(plot_data):,} of "
            f"{len(dataframe):,} rows."
        )

    def add_chart(title: str, description: str, figure: Figure) -> None:
        """Append a plot while respecting the report chart limit."""
        if len(report.charts) < MAX_CHARTS:
            report.charts.append(VisualizationCard(title, description, figure))
        else:
            plt.close(figure)

    missing = calculate_missing_values(dataframe)
    if (missing["Missing Count"] > 0).any():
        add_chart(
            "Missing values by column",
            "Identifies fields that may need cleaning before further analysis.",
            missingness_bar_chart(missing),
        )

    dated = [column for column, kind in types.items() if kind == "datetime"]
    numeric = _numeric_columns(dataframe, types)
    if dated and numeric:
        date_column = dated[0]
        measure = next(
            (column for column in numeric if types[column] == "numeric_continuous"),
            numeric[0],
        )
        add_chart(
            f"{measure} over {date_column}",
            "A time trend can reveal changes, seasonality, or unusual periods.",
            data_story_lineplot(plot_data, date_column, measure),
        )

    numeric_continuous = [
        column for column in numeric if types[column] == "numeric_continuous"
    ]
    for column in numeric_continuous[:1]:
        add_chart(
            f"Distribution of {column}",
            "Shows the typical range, shape, and possible extreme values.",
            data_story_histogram(plot_data, column),
        )

    if len(numeric) >= 2:
        heatmap_columns = numeric[:MAX_HEATMAP_COLUMNS]
        add_chart(
            "Numeric relationships",
            "Correlation summarizes linear association, not cause and effect.",
            data_story_heatmap(dataframe[heatmap_columns]),
        )
        correlations = dataframe[numeric].apply(
            pd.to_numeric, errors="coerce"
        ).corr()
        upper = correlations.where(
            np.triu(np.ones(correlations.shape), k=1).astype(bool)
        ).stack().dropna()
        if not upper.empty:
            pair = upper.abs().idxmax()
            if (
                abs(float(correlations.loc[pair[0], pair[1]])) >= 0.3
                and types[pair[0]] == "numeric_continuous"
                and types[pair[1]] == "numeric_continuous"
            ):
                add_chart(
                    f"{pair[1]} vs {pair[0]}",
                    "The scatterplot helps inspect the strongest numeric association.",
                    data_story_scatter(plot_data, pair[0], pair[1]),
                )

    categories = [
        column for column, kind in types.items()
        if kind in {"categorical", "binary", "numeric_discrete"}
        and 1 < dataframe[column].nunique(dropna=True) <= MAX_CATEGORY_LEVELS
    ]
    category = categories[0] if categories else None
    measure = next(
        (column for column in numeric_continuous if column not in {category}),
        None,
    )
    if category and measure:
        add_chart(
            f"{measure} by {category}",
            "Compare the numeric distribution across the groups in this column.",
            data_story_boxplot(plot_data, measure, category),
        )

    if category:
        add_chart(
            f"Most common {category} values",
            "Shows the most frequent groups and highlights category imbalance.",
            category_bar(dataframe, category),
        )

    if not report.charts:
        report.notices.append(
            "No informative chart could be selected; the file may contain only "
            "constant, empty, or identifier-like columns."
        )
    return report
