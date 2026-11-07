"""Matplotlib and seaborn charts for automatic dataset exploration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype

from src.type_inference import infer_column_type

MAX_SCATTER_POINTS = 20_000


def _finish_plot(
    figure: Figure,
    axis: object,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Figure:
    """Set consistent plot labels and fit the chart to its canvas."""
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    figure.tight_layout()
    return figure


def _require_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a requested column or raise a clear validation error."""
    if column not in dataframe.columns:
        raise ValueError(f"Column {column!r} is not present in the dataset.")
    return dataframe[column]


def _paired_data(
    dataframe: pd.DataFrame,
    columns: list[str],
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Copy plotted columns, convert numeric inputs, and drop incomplete rows."""
    working = dataframe.loc[:, columns].copy()
    for column in numeric_columns or []:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.dropna(subset=columns)


def data_story_histogram(dataframe: pd.DataFrame, column: str) -> Figure:
    """Draw a labeled histogram with KDE and mean/median reference lines."""
    values = pd.to_numeric(_require_column(dataframe, column), errors="coerce").dropna()
    figure, axis = plt.subplots(figsize=(9, 5))
    if values.empty:
        axis.text(
            0.5, 0.5, f"No numeric values in {column}", ha="center",
            va="center", transform=axis.transAxes,
        )
    else:
        sns.histplot(values, kde=len(values) > 1, ax=axis)
        axis.axvline(
            values.mean(), color="#C00000", linestyle="--",
            label=f"Mean: {values.mean():.3g}",
        )
        axis.axvline(
            values.median(), color="#70AD47", linestyle=":",
            label=f"Median: {values.median():.3g}",
        )
        axis.legend()
    return _finish_plot(figure, axis, f"Distribution of {column}", column, "Count")


def data_story_scatter(dataframe: pd.DataFrame, x: str, y: str) -> Figure:
    """Draw numeric columns as points with a fitted regression line."""
    working = _paired_data(dataframe, [x, y], [x, y])
    figure, axis = plt.subplots(figsize=(9, 5))
    if len(working) >= 2 and working[x].nunique() > 1 and working[y].nunique() > 1:
        sns.regplot(data=working, x=x, y=y, ax=axis, scatter_kws={"alpha": 0.65})
    else:
        sns.scatterplot(data=working, x=x, y=y, ax=axis)
    return _finish_plot(figure, axis, f"{y} vs {x}", x, y)


def data_story_boxplot(
    dataframe: pd.DataFrame, numeric: str, category: str
) -> Figure:
    """Compare a numeric distribution across the most common groups."""
    working = _paired_data(dataframe, [numeric, category], [numeric])
    figure, axis = plt.subplots(figsize=(9, 5))
    if working.empty:
        axis.text(
            0.5, 0.5, "No complete values to plot", ha="center",
            va="center", transform=axis.transAxes,
        )
    else:
        labels = working[category].astype(str).value_counts().head(20).index
        working = working[working[category].astype(str).isin(labels)].copy()
        grouped_values = [
            working.loc[working[category].astype(str) == label, numeric].to_numpy()
            for label in labels
        ]
        axis.boxplot(
            grouped_values,
            tick_labels=list(labels),
            orientation="vertical",
            patch_artist=True,
            boxprops={"facecolor": "#70AD47"},
        )
        if len(working) < 500:
            for position, label in enumerate(labels):
                values = working.loc[
                    working[category].astype(str) == label, numeric
                ].to_numpy()
                jitter = np.linspace(-0.12, 0.12, len(values))
                axis.scatter(
                    position + jitter, values, color="black", alpha=0.35,
                    s=12, zorder=3,
                )
        axis.tick_params(axis="x", labelrotation=30)
    return _finish_plot(figure, axis, f"{numeric} by {category}", category, numeric)


def data_story_lineplot(dataframe: pd.DataFrame, x: str, y: str) -> Figure:
    """Draw a numeric measure over ordered dates with a rolling mean."""
    working = dataframe.loc[:, [x, y]].copy()
    datetime_axis = infer_column_type(working[x], x) == "datetime"
    if datetime_axis:
        working[x] = pd.to_datetime(working[x], errors="coerce", format="mixed")
    working[y] = pd.to_numeric(working[y], errors="coerce")
    working = working.dropna(subset=[x, y]).sort_values(x)
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=working, x=x, y=y, marker="o", ax=axis)
    if datetime_axis and len(working) >= 3:
        time_values = (
            working.groupby(x, as_index=False, observed=True)[y]
            .mean()
            .sort_values(x)
        )
        window = min(3, len(time_values))
        rolling = time_values[y].rolling(window=window, min_periods=1).mean()
        axis.plot(
            time_values[x], rolling, color="#C00000", linestyle="--",
            label=f"{window}-point rolling mean",
        )
        axis.legend()
    axis.tick_params(axis="x", labelrotation=30)
    return _finish_plot(figure, axis, f"{y} by {x}", x, y)


def data_story_heatmap(dataframe: pd.DataFrame) -> Figure:
    """Draw pairwise numeric correlations, including numeric-string columns."""
    numeric = dataframe.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    figure, axis = plt.subplots(figsize=(8, 6))
    if numeric.shape[1] < 2:
        axis.text(
            0.5, 0.5, "At least two numeric columns are needed",
            ha="center", va="center", transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        sns.heatmap(
            numeric.corr(), annot=True, cmap="coolwarm", center=0,
            fmt=".2f", ax=axis,
        )
        axis.set_title("Numeric feature correlations")
    figure.tight_layout()
    return figure


def missingness_bar_chart(missing_values: pd.DataFrame) -> Figure:
    """Build a bar chart for columns that contain missing values."""
    data = missing_values[missing_values["Missing Count"] > 0]
    figure, axis = plt.subplots(figsize=(8, 4))
    if data.empty:
        axis.text(
            0.5, 0.5, "No missing values detected",
            ha="center", va="center", transform=axis.transAxes,
        )
    else:
        sns.barplot(data=data, x="Missing %", y="Column", ax=axis, color="#ED7D31")
        axis.axvline(30, color="#C00000", linestyle="--", linewidth=1)
    axis.set(title="Missing values by column", xlabel="Missing (%)", ylabel="")
    figure.tight_layout()
    return figure


def category_bar(dataframe: pd.DataFrame, column: str, top_n: int = 20) -> Figure:
    """Build a bar chart for the most frequent category values."""
    series = _require_column(dataframe, column)
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    counts = series.dropna().astype(str).value_counts().head(top_n)
    figure, axis = plt.subplots(figsize=(9, 4))
    if counts.empty:
        axis.text(
            0.5, 0.5, "No non-missing categories to plot",
            ha="center", va="center", transform=axis.transAxes,
        )
    else:
        sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#70AD47")
    axis.set(title=f"Top categories in {column}", xlabel="Count", ylabel=column)
    figure.tight_layout()
    return figure


def histogram(dataframe: pd.DataFrame, column: str) -> Figure:
    """Build a reusable histogram for a numeric column."""
    series = _require_column(dataframe, column)
    if not is_numeric_dtype(series):
        raise ValueError(f"Column {column!r} is not numeric.")
    return data_story_histogram(dataframe, column)


def correlation_heatmap(dataframe: pd.DataFrame) -> Figure:
    """Build a reusable correlation heatmap."""
    return data_story_heatmap(dataframe)


def boxplot(dataframe: pd.DataFrame, column: str) -> Figure:
    """Build a reusable boxplot for a numeric column."""
    series = _require_column(dataframe, column)
    if not is_numeric_dtype(series):
        raise ValueError(f"Column {column!r} is not numeric.")
    values = series.dropna()
    figure, axis = plt.subplots(figsize=(7, 4))
    if values.empty:
        axis.text(
            0.5, 0.5, "No non-missing values to plot",
            ha="center", va="center", transform=axis.transAxes,
        )
    else:
        axis.boxplot(
            values, orientation="horizontal", patch_artist=True,
            boxprops={"facecolor": "#ED7D31"},
        )
    axis.set(title=f"Boxplot of {column}", xlabel=column)
    figure.tight_layout()
    return figure
