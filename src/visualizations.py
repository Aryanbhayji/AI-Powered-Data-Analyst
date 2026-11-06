"""Matplotlib and seaborn chart builders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype

from src.type_inference import infer_column_type, infer_column_types


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


def _paired_data(
    dataframe: pd.DataFrame, columns: list[str], numeric_columns: list[str] | None = None
) -> pd.DataFrame:
    """Copy only plotted columns, convert numeric inputs, and drop incomplete rows."""
    working = dataframe.loc[:, columns].copy()
    for column in numeric_columns or []:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.dropna(subset=columns)


def command_histogram(dataframe: pd.DataFrame, column: str) -> Figure:
    """Draw a labeled histogram with a KDE when at least two values exist."""
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    figure, axis = plt.subplots(figsize=(9, 5))
    if values.empty:
        axis.text(0.5, 0.5, f"No numeric values in {column}", ha="center",
                  va="center", transform=axis.transAxes)
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


def command_scatter(dataframe: pd.DataFrame, x: str, y: str) -> Figure:
    """Draw numeric columns as points with a fitted regression line."""
    working = _paired_data(dataframe, [x, y], [x, y])
    figure, axis = plt.subplots(figsize=(9, 5))
    if len(working) >= 2 and working[x].nunique() > 1 and working[y].nunique() > 1:
        sns.regplot(data=working, x=x, y=y, ax=axis, scatter_kws={"alpha": 0.65})
    else:
        sns.scatterplot(data=working, x=x, y=y, ax=axis)
    return _finish_plot(figure, axis, f"{y} vs {x}", x, y)


def command_boxplot(dataframe: pd.DataFrame, numeric: str, category: str) -> Figure:
    """Draw a numeric distribution grouped by a categorical or binary column."""
    working = _paired_data(dataframe, [numeric, category], [numeric])
    figure, axis = plt.subplots(figsize=(9, 5))
    if working.empty:
        axis.text(0.5, 0.5, "No complete values to plot", ha="center",
                  va="center", transform=axis.transAxes)
    else:
        categories = working[category].astype(str).value_counts().head(20).index
        working = working[working[category].astype(str).isin(categories)].copy()
        labels = list(categories)
        grouped_values = [
            working.loc[working[category].astype(str) == label, numeric].to_numpy()
            for label in labels
        ]
        axis.boxplot(
            grouped_values,
            tick_labels=labels,
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
                    position + jitter, values, color="black",
                    alpha=0.35, s=12, zorder=3,
                )
        axis.tick_params(axis="x", labelrotation=30)
    return _finish_plot(figure, axis, f"{numeric} by {category}", category, numeric)


def command_countplot(
    dataframe: pd.DataFrame, category: str, hue: str | None = None
) -> Figure:
    """Draw category counts with an optional grouped hue."""
    columns = [category] + ([hue] if hue and hue != category else [])
    working = _paired_data(dataframe, columns)
    figure, axis = plt.subplots(figsize=(9, 5))
    if working.empty:
        axis.text(0.5, 0.5, "No complete categories to plot", ha="center",
                  va="center", transform=axis.transAxes)
    else:
        top = working[category].astype(str).value_counts().head(20).index
        working = working[working[category].astype(str).isin(top)].copy()
        sns.countplot(data=working, x=category, hue=hue, ax=axis)
        axis.tick_params(axis="x", labelrotation=30)
    return _finish_plot(
        figure, axis,
        f"Counts of {category}" + (f" by {hue}" if hue else ""),
        category, "Count",
    )


def command_rate_bar(
    dataframe: pd.DataFrame, outcome: str, category: str
) -> Figure:
    """Draw the mean binary outcome across category levels."""
    working = _paired_data(dataframe, [outcome, category], [outcome])
    figure, axis = plt.subplots(figsize=(9, 5))
    if working.empty:
        axis.text(0.5, 0.5, "No complete values to plot", ha="center",
                  va="center", transform=axis.transAxes)
    else:
        working[category] = working[category].astype(str)
        top = working[category].value_counts().head(20).index
        working = working[working[category].isin(top)]
        sns.barplot(data=working, x=category, y=outcome, errorbar=None, ax=axis)
        axis.tick_params(axis="x", labelrotation=30)
    return _finish_plot(
        figure, axis, f"Mean {outcome} by {category}", category, f"Mean {outcome}"
    )


def command_lineplot(dataframe: pd.DataFrame, x: str, y: str) -> Figure:
    """Draw a numeric measure over an ordered or datetime x-axis."""
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
        time_values["rolling_mean"] = time_values[y].rolling(
            window=min(3, len(time_values)), min_periods=1
        ).mean()
        axis.plot(
            time_values[x],
            time_values["rolling_mean"],
            color="#C00000",
            linestyle="--",
            label="3-point rolling mean",
        )
        axis.legend()
    axis.tick_params(axis="x", labelrotation=30)
    return _finish_plot(figure, axis, f"{y} by {x}", x, y)


def command_pie(dataframe: pd.DataFrame, category: str, top_n: int = 20) -> Figure:
    """Draw a bounded category pie chart."""
    counts = dataframe[category].dropna().astype(str).value_counts()
    shown = counts.head(min(top_n, 20))
    if len(counts) > len(shown):
        shown.loc["Other"] = counts.iloc[len(shown):].sum()
    figure, axis = plt.subplots(figsize=(9, 5))
    if shown.empty:
        axis.text(0.5, 0.5, "No categories to plot", ha="center",
                  va="center", transform=axis.transAxes)
    else:
        axis.pie(shown.values, labels=shown.index, autopct="%1.1f%%")
    return _finish_plot(
        figure, axis, f"Distribution of {category}", category, "Share"
    )


def command_heatmap(dataframe: pd.DataFrame) -> Figure:
    """Draw a heatmap of available numeric correlations."""
    numeric = dataframe.select_dtypes(include="number")
    if numeric.empty:
        numeric = dataframe.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    figure, axis = plt.subplots(figsize=(9, 5))
    if numeric.shape[1] >= 2:
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, ax=axis)
    else:
        axis.text(0.5, 0.5, "At least two numeric columns are needed",
                  ha="center", va="center", transform=axis.transAxes)
    return _finish_plot(
        figure, axis, "Numeric correlation heatmap", "Feature", "Feature"
    )


def command_violinplot(dataframe: pd.DataFrame, numeric: str, category: str) -> Figure:
    """Draw a numeric distribution by category as a violin plot."""
    working = _paired_data(dataframe, [numeric, category], [numeric])
    figure, axis = plt.subplots(figsize=(9, 5))
    if not working.empty:
        working[category] = working[category].astype(str)
        sns.violinplot(data=working, x=category, y=numeric, cut=0, ax=axis)
        axis.tick_params(axis="x", labelrotation=30)
    else:
        axis.text(0.5, 0.5, "No complete values to plot", ha="center",
                  va="center", transform=axis.transAxes)
    return _finish_plot(figure, axis, f"{numeric} by {category}", category, numeric)


def command_pairplot(dataframe: pd.DataFrame) -> Figure:
    """Draw a compact pairplot for up to five numeric columns."""
    numeric = dataframe.select_dtypes(include="number").copy()
    if numeric.shape[1] < 2:
        numeric = dataframe.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.iloc[:, :5].dropna()
    if numeric.shape[1] < 2 or numeric.empty:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.text(0.5, 0.5, "At least two usable numeric columns are needed",
                  ha="center", va="center", transform=axis.transAxes)
        return _finish_plot(figure, axis, "Numeric pair plot", "", "")
    grid = sns.pairplot(numeric)
    grid.fig.set_size_inches(9, 5)
    grid.fig.suptitle("Numeric pair plot")
    if grid.fig.axes:
        grid.fig.axes[0].set_title("Pairwise relationships")
    grid.fig.tight_layout()
    return grid.fig


def create_command_plot(
    dataframe: pd.DataFrame,
    plan: dict[str, object],
    command: str = "",
) -> tuple[Figure, str, int]:
    """Repair plot intent, choose a compatible chart, and always return a figure."""
    types = infer_column_types(dataframe)
    command_text = command.casefold()
    requested = str(plan.get("plot_type") or "auto").casefold()
    aliases = {
        "boxplot": "box", "box plot": "box", "count plot": "count",
        "countplot": "count", "heat map": "heatmap",
    }
    requested = aliases.get(requested, requested)
    x = plan.get("x")
    y = plan.get("y")
    hue = plan.get("hue") or plan.get("group_by")
    left = plan.get("var_a") or x
    right = plan.get("var_b") or y
    if plan.get("action") in {"relationship", "correlate"}:
        x, y = left, right
    if requested == "auto":
        for phrase, chart in (
            ("histogram", "histogram"), ("scatter", "scatter"),
            ("boxplot", "box"), ("box plot", "box"), ("violin", "violin"),
            ("heatmap", "heatmap"), ("heat map", "heatmap"),
            ("pie", "pie"), ("count plot", "count"), ("countplot", "count"),
            ("line chart", "line"), ("line plot", "line"),
        ):
            if phrase in command_text:
                requested = chart
                break

    if "correlation" in command_text and x is None and y is None:
        requested = "heatmap"
    if requested in {"auto", "unsupported", "none"}:
        requested = "auto"
    valid_columns = set(str(column) for column in dataframe.columns)
    if x not in valid_columns:
        x = None
    if y not in valid_columns:
        y = None
    if hue not in valid_columns:
        hue = None
    if x is None and y is None and isinstance(left, str) and left in valid_columns:
        x = left
    if x is None and y is None and isinstance(right, str) and right in valid_columns:
        x = right

    numeric = {name for name, kind in types.items()
               if kind in {"numeric_continuous", "numeric_discrete", "binary"}}
    histogram_columns = {
        name for name, kind in types.items()
        if kind in {"numeric_continuous", "numeric_discrete"}
    }
    relationship = plan.get("action") in {"relationship", "correlate"}
    if requested == "auto":
        if x and y:
            x_type, y_type = types.get(x), types.get(y)
            if x_type == "datetime" and y_type in {
                "binary", "numeric_discrete", "numeric_continuous"
            }:
                requested = "line"
            elif y_type == "datetime" and x_type in {
                "binary", "numeric_discrete", "numeric_continuous"
            }:
                x, y = y, x
                requested = "line"
            elif "rate" in command_text and "binary" in {x_type, y_type}:
                requested = "bar"
            elif "binary" in {x_type, y_type} and (
                x_type in {"categorical", "numeric_discrete"}
                or y_type in {"categorical", "numeric_discrete"}
            ):
                requested = "count"
            elif x_type == "numeric_continuous" and y_type == "numeric_continuous":
                requested = "scatter"
            elif x_type == "numeric_continuous" and y_type in {
                "binary", "numeric_discrete", "categorical"
            }:
                requested = "box"
            elif y_type == "numeric_continuous" and x_type in {
                "binary", "numeric_discrete", "categorical"
            }:
                requested = "box"
            else:
                requested = "count"
        elif x or y:
            single = x or y
            requested = "histogram" if single in histogram_columns else "count"
        else:
            requested = "heatmap" if "correlation" in command_text else "histogram"

    warning = ""
    requested_columns = [column for column in (x, y, hue) if column]
    if requested == "heatmap":
        figure = command_heatmap(dataframe)
        used = len(dataframe)
    elif requested == "pair":
        figure = command_pairplot(dataframe)
        used = len(dataframe)
    elif requested == "histogram":
        column = x or y or next(iter(numeric), dataframe.columns[0] if len(dataframe.columns) else "")
        if column not in dataframe:
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.text(0.5, 0.5, "No columns are available to plot", ha="center")
            figure = _finish_plot(figure, axis, "Dataset plot", "", "")
            return figure, warning, 0
        if column not in histogram_columns and not dataframe[column].isna().all():
            warning = f"Histogram needs numeric data; I used a count plot for {column} instead."
            figure = command_countplot(dataframe, column)
            used = int(dataframe[column].notna().sum())
        else:
            figure = command_histogram(dataframe, column)
            used = int(pd.to_numeric(dataframe[column], errors="coerce").notna().sum())
    elif requested in {"scatter", "line"}:
        if not (x and y):
            requested = "auto"
            return create_command_plot(dataframe, {**plan, "plot_type": "auto"}, command)
        if x not in numeric or y not in numeric:
            warning = "A scatter plot needs two numeric columns; I used the closest compatible chart."
            if x in numeric:
                figure = command_boxplot(dataframe, x, y)
            elif y in numeric:
                figure = command_boxplot(dataframe, y, x)
            else:
                figure = command_countplot(dataframe, x, y)
        else:
            plot_data = dataframe
            if requested == "scatter":
                paired = _paired_data(dataframe, [x, y], [x, y])
                if len(paired) > 20_000:
                    plot_data = paired.sample(20_000, random_state=0)
                    warning = "Showing a deterministic sample of 20,000 rows."
                figure = command_scatter(plot_data, x, y)
            else:
                figure = command_lineplot(dataframe, x, y)
        used = len(_paired_data(dataframe, [x, y], [name for name in (x, y) if name in numeric]))
    elif requested in {"box", "violin"}:
        number = (
            x if types.get(x) == "numeric_continuous"
            else y if types.get(y) == "numeric_continuous"
            else None
        )
        category = (
            y if number == x and y != number
            else x if number == y and x != number
            else hue
        )
        if number and category and types.get(category) in {
            "binary", "numeric_discrete", "categorical"
        }:
            figure = command_boxplot(dataframe, number, category) if requested == "box" else command_violinplot(dataframe, number, category)
            used = len(_paired_data(dataframe, [number, category], [number]))
        elif number:
            warning = "A grouped boxplot needs a category; I used a histogram instead."
            figure = command_histogram(dataframe, number)
            used = int(pd.to_numeric(dataframe[number], errors="coerce").notna().sum())
        else:
            warning = "A boxplot needs numeric data; I used a count plot instead."
            category = x or y or next(iter(dataframe.columns), "")
            figure = command_countplot(dataframe, category) if category in dataframe else command_heatmap(dataframe)
            used = int(dataframe[category].notna().sum()) if category in dataframe else len(dataframe)
    elif requested == "pie":
        category = x or y or next(iter(types), "")
        figure = command_pie(dataframe, category) if category in dataframe else command_heatmap(dataframe)
        used = int(dataframe[category].notna().sum()) if category in dataframe else len(dataframe)
    elif requested == "bar" and (x or y):
        first, second = x or y, y if x else None
        if second and types.get(first) == "binary" and types.get(second) in {
            "categorical", "numeric_discrete"
        }:
            figure = command_rate_bar(dataframe, first, second)
            used = len(_paired_data(dataframe, [first, second], [first]))
        elif second and types.get(second) == "binary" and types.get(first) in {
            "categorical", "numeric_discrete"
        }:
            figure = command_rate_bar(dataframe, second, first)
            used = len(_paired_data(dataframe, [first, second], [second]))
        elif second and second in numeric:
            figure = command_rate_bar(dataframe, second, first)
            used = len(_paired_data(dataframe, [first, second], [second]))
        else:
            figure = command_countplot(dataframe, first)
            used = int(dataframe[first].notna().sum())
    elif requested == "count":
        if x and y and types.get(x) in {
            "binary", "categorical", "numeric_discrete"
        } and types.get(y) in {"binary", "categorical", "numeric_discrete"}:
            category, count_hue = x, y
        elif x and y and types.get(y) in {
            "binary", "categorical", "numeric_discrete"
        } and types.get(x) in {"binary", "categorical", "numeric_discrete"}:
            category, count_hue = y, x
        else:
            category, count_hue = x or y or next(iter(types), ""), None
        figure = (
            command_countplot(dataframe, category, count_hue)
            if category in dataframe else command_heatmap(dataframe)
        )
        used = int(dataframe[category].notna().sum()) if category in dataframe else len(dataframe)
    else:
        fallback = x or next(iter(numeric), None)
        if fallback in dataframe and fallback in numeric:
            warning = f"I used a histogram of {fallback} as the closest available plot."
            figure = command_histogram(dataframe, fallback)
            used = int(pd.to_numeric(dataframe[fallback], errors="coerce").notna().sum())
        elif len(dataframe.columns):
            fallback = str(dataframe.columns[0])
            warning = f"I used a count plot of {fallback} as the closest available plot."
            figure = command_countplot(dataframe, fallback)
            used = int(dataframe[fallback].notna().sum())
        else:
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.text(0.5, 0.5, "No columns are available to plot", ha="center")
            figure = _finish_plot(figure, axis, "Dataset plot", "", "")
            used = 0
    if relationship and not requested_columns:
        used = len(dataframe)
    return figure, warning, used

def _require_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a requested column or raise a clear validation error."""
    if column not in dataframe.columns:
        raise ValueError(f"Column {column!r} is not present in the dataset.")
    return dataframe[column]


def missingness_bar_chart(missing_values: pd.DataFrame) -> Figure:
    """Build a bar chart for columns that contain missing values."""
    data = missing_values[missing_values["Missing Count"] > 0]
    figure, axis = plt.subplots(figsize=(8, 4))
    if data.empty:
        axis.text(0.5, 0.5, "No missing values detected",
                  ha="center", va="center", transform=axis.transAxes)
    else:
        sns.barplot(data=data, x="Missing %", y="Column", ax=axis, color="#ED7D31")
        axis.axvline(30, color="#C00000", linestyle="--", linewidth=1)
    axis.set(title="Missing values by column", xlabel="Missing (%)", ylabel="")
    figure.tight_layout()
    return figure


def histogram(dataframe: pd.DataFrame, column: str) -> Figure:
    """Build a histogram for a numeric column."""
    series = _require_column(dataframe, column)
    if not is_numeric_dtype(series):
        raise ValueError(f"Column {column!r} is not numeric.")
    figure, axis = plt.subplots(figsize=(8, 4))
    values = series.dropna()
    if values.empty:
        axis.text(0.5, 0.5, "No non-missing values to plot",
                  ha="center", va="center", transform=axis.transAxes)
    else:
        sns.histplot(values, kde=len(values) > 1, ax=axis, color="#4472C4")
    axis.set(title=f"Distribution of {column}", xlabel=column, ylabel="Count")
    figure.tight_layout()
    return figure


def category_bar(dataframe: pd.DataFrame, column: str, top_n: int = 20) -> Figure:
    """Build a bar chart for up to ``top_n`` most frequent category values."""
    series = _require_column(dataframe, column)
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    counts = series.dropna().astype(str).value_counts().head(top_n)
    figure, axis = plt.subplots(figsize=(9, 4))
    if counts.empty:
        axis.text(0.5, 0.5, "No non-missing categories to plot",
                  ha="center", va="center", transform=axis.transAxes)
    else:
        sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#70AD47")
    axis.set(title=f"Top categories in {column}", xlabel="Count", ylabel=column)
    figure.tight_layout()
    return figure


def correlation_heatmap(dataframe: pd.DataFrame) -> Figure:
    """Build a correlation heatmap when the data has multiple numeric columns."""
    numeric = dataframe.select_dtypes(include="number")
    figure, axis = plt.subplots(figsize=(8, 6))
    if numeric.shape[1] < 2:
        axis.text(0.5, 0.5, "At least two numeric columns are needed",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0,
                    fmt=".2f", ax=axis)
        axis.set_title("Numeric feature correlations")
    figure.tight_layout()
    return figure


def boxplot(dataframe: pd.DataFrame, column: str) -> Figure:
    """Build a boxplot for a numeric column."""
    series = _require_column(dataframe, column)
    if not is_numeric_dtype(series):
        raise ValueError(f"Column {column!r} is not numeric.")
    figure, axis = plt.subplots(figsize=(7, 4))
    values = series.dropna()
    if values.empty:
        axis.text(0.5, 0.5, "No non-missing values to plot",
                  ha="center", va="center", transform=axis.transAxes)
    else:
        axis.boxplot(
            values,
            orientation="horizontal",
            patch_artist=True,
            boxprops={"facecolor": "#ED7D31"},
        )
    axis.set(title=f"Boxplot of {column}", xlabel=column)
    figure.tight_layout()
    return figure
