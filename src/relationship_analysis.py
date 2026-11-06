"""Type-aware statistical relationship analysis and visualizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

from src.type_inference import infer_column_type, infer_column_types

NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete", "binary"}
CATEGORY_TYPES = {"binary", "numeric_discrete", "categorical", "text_id"}


@dataclass
class RelationshipResult:
    """All charts, statistical output, and narrative for one relationship."""

    charts: dict[str, Figure] = field(default_factory=dict)
    statistics: pd.DataFrame = field(default_factory=pd.DataFrame)
    explanation: str = ""
    summary: str = ""
    warnings: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def _numeric(series: pd.Series) -> pd.Series:
    """Convert numeric-looking values without modifying source data."""
    return pd.to_numeric(series.copy(), errors="coerce")


def _figure(title: str) -> tuple[Figure, Any]:
    """Create a consistently styled chart canvas."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.set_title(title)
    figure.tight_layout()
    return figure, axis


def _stats_table(values: dict[str, Any]) -> pd.DataFrame:
    """Represent scalar computed results as a downloadable table."""
    return pd.DataFrame(
        [{"Statistic": key, "Value": value} for key, value in values.items()]
    )


def _safe_p(value: float) -> str:
    """Format p-values without implying more precision than the test supports."""
    return f"{value:.4g}" if np.isfinite(value) else "not available"


def _explain_numeric_group(
    metric: str,
    category: str,
    grouped: pd.DataFrame,
    test_name: str,
    p_value: float,
    effect: float | None,
    binary_outcome: bool = False,
) -> str:
    """Describe group differences using only computed aggregate statistics."""
    medians = grouped.groupby(category, observed=True)[metric].median().sort_values()
    if len(medians) >= 2:
        low_name, high_name = str(medians.index[0]), str(medians.index[-1])
        low, high = float(medians.iloc[0]), float(medians.iloc[-1])
        claim = (
            f"{metric} has a median of {high:.3g} for {high_name}, versus "
            f"{low:.3g} for {low_name}."
        )
    else:
        claim = f"The median {metric} is {float(medians.iloc[0]):.3g}."
    significance = (
        "The difference is statistically significant"
        if np.isfinite(p_value) and p_value < 0.05
        else "There is not strong statistical evidence of a difference"
    )
    suffix = f" ({test_name} p = {_safe_p(p_value)})."
    if effect is not None and np.isfinite(effect):
        suffix += f" Effect size: {effect:.3g}."
    outcome_note = (
        " Group rates can differ, but this analysis does not establish cause."
        if binary_outcome else ""
    )
    return (
        f"{claim} {significance}{suffix}{outcome_note} Correlation and group "
        "differences do not establish causation; correlation is not causation."
    )


def _numeric_categorical(
    data: pd.DataFrame, numeric: str, category: str
) -> RelationshipResult:
    """Compare a numeric measure across groups with tests and four charts."""
    working = data[[numeric, category]].copy()
    working[numeric] = _numeric(working[numeric])
    working = working.dropna()
    if len(working) < 5:
        raise ValueError(
            "Fewer than 5 rows have both values. Choose columns with more complete data."
        )
    working[category] = working[category].astype(str)
    top_groups = working[category].value_counts().head(19).index
    working[category] = working[category].where(
        working[category].isin(top_groups), "Other"
    )
    groups = [
        values[numeric].to_numpy()
        for _, values in working.groupby(category, observed=True, sort=False)
        if len(values)
    ]
    if len(groups) < 2:
        raise ValueError("At least two groups with usable values are needed.")

    if len(groups) == 2:
        test = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        test_name, p_value = "Mann–Whitney U", float(test.pvalue)
        pooled = np.sqrt((np.var(groups[0], ddof=1) + np.var(groups[1], ddof=1)) / 2)
        effect = (
            float((np.mean(groups[1]) - np.mean(groups[0])) / pooled)
            if pooled > 0 else 0.0
        )
    else:
        test = stats.kruskal(*groups)
        test_name, p_value = "Kruskal–Wallis", float(test.pvalue)
        grand_mean = float(working[numeric].mean())
        between = sum(
            len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups
        )
        total_ss = float(((working[numeric] - grand_mean) ** 2).sum())
        effect = between / total_ss if total_ss > 0 else 0.0

    summary = working.groupby(category, observed=True)[numeric].agg(
        ["count", "mean", "median", "std"]
    ).reset_index()
    rows_used = len(working)
    result_stats: dict[str, Any] = {
        "Analyzed columns": f"{numeric} by {category}",
        "Rows used": rows_used,
        "Rows dropped for missing values": len(data) - rows_used,
        "Groups analyzed": len(groups),
        "Test": test_name,
        "p-value": p_value,
        "Effect size": effect,
    }
    skew = float(working[numeric].skew())
    use_log_scale = (
        np.isfinite(skew)
        and abs(skew) > 1
        and bool((working[numeric] > 0).all())
    )
    result_stats["Skewness"] = skew
    result_stats["Boxplot uses log scale"] = use_log_scale
    for _, row in summary.iterrows():
        label = f"{category}={row[category]}"
        for field_name in ("count", "mean", "median", "std"):
            result_stats[f"{label} {field_name}"] = row[field_name]

    binary_outcome = infer_column_type(data[category], category) == "binary"
    if binary_outcome:
        binary = _numeric(data[category])
        paired = pd.DataFrame({numeric: _numeric(data[numeric]), category: binary}).dropna()
        if paired[category].nunique() == 2 and paired[numeric].nunique() > 1:
            point_biserial = stats.pointbiserialr(paired[category], paired[numeric])
            result_stats["Point-biserial r"] = float(point_biserial.statistic)
            result_stats["Point-biserial p-value"] = float(point_biserial.pvalue)
            bins = pd.qcut(
                paired[numeric], q=min(5, paired[numeric].nunique()),
                duplicates="drop",
            )
            rate = paired.groupby(bins, observed=True)[category].mean()
            fig, ax = _figure(f"{category} rate by {numeric} quantile")
            rate.plot(kind="bar", ax=ax, color="#4472C4")
            ax.set_ylabel(f"Mean {category} rate")
            ax.tick_params(axis="x", labelrotation=25)
            result_stats["Outcome-rate bins"] = len(rate)
            rate_figure = fig
        else:
            rate_figure = _empty_chart(f"{category} rate by {numeric}", "No valid binary outcome")
    else:
        rate_figure = _empty_chart(f"{category} rate by {numeric}", "Outcome column is not binary")

    box, box_ax = _figure(f"{numeric} by {category}")
    box_groups = working.groupby(category, observed=True, sort=True)
    box_ax.boxplot(
        [values[numeric].to_numpy() for _, values in box_groups],
        tick_labels=[str(name) for name, _ in box_groups],
        showfliers=True,
        orientation="vertical",
        patch_artist=True,
        boxprops={"facecolor": "#70AD47"},
    )
    if use_log_scale:
        box_ax.set_yscale("log")
    box_ax.tick_params(axis="x", labelrotation=30)
    violin, violin_ax = _figure(f"{numeric} distributions by {category}")
    if working[category].nunique() <= 10:
        sns.violinplot(data=working, x=category, y=numeric, ax=violin_ax, cut=0)
    else:
        sns.kdeplot(data=working, x=numeric, hue=category, common_norm=False, ax=violin_ax)
    violin_ax.tick_params(axis="x", labelrotation=30)
    grouped = working.groupby(category, observed=True)[numeric]
    means = grouped.mean()
    errors = 1.96 * grouped.sem().fillna(0)
    mean_fig, mean_ax = _figure(f"Mean {numeric} by {category} (95% CI)")
    mean_ax.bar(means.index.astype(str), means.values, yerr=errors.values, capsize=4)
    mean_ax.tick_params(axis="x", labelrotation=30)
    result = RelationshipResult(
        charts={
            "Boxplot": box, "Group distributions": violin,
            "Group means with 95% CI": mean_fig, "Outcome rate by quantile": rate_figure,
        },
        statistics=_stats_table(result_stats),
        tables={"Group-wise summary": summary},
        explanation=_explain_numeric_group(
            numeric, category, working, test_name, p_value, effect, binary_outcome
        ),
    )
    result.summary = result.explanation
    return result


def _empty_chart(title: str, message: str) -> Figure:
    """Render a friendly note instead of failing on degenerate data."""
    figure, axis = _figure(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    return figure


def _numeric_numeric(data: pd.DataFrame, left: str, right: str) -> RelationshipResult:
    """Compare two numeric columns with correlations and binned trends."""
    working = pd.DataFrame({left: _numeric(data[left]), right: _numeric(data[right])}).dropna()
    if len(working) < 5 or working[left].nunique() < 2 or working[right].nunique() < 2:
        raise ValueError("At least 5 paired rows with varying values are needed.")
    pearson = stats.pearsonr(working[left], working[right])
    spearman = stats.spearmanr(working[left], working[right])
    charts: dict[str, Figure] = {}
    figure, axis = _figure(f"{left} and {right}")
    if len(working) > 5000:
        axis.hexbin(working[left], working[right], gridsize=35, cmap="Blues", mincnt=1)
        axis.set_ylabel(right)
        axis.set_xlabel(left)
    else:
        sns.regplot(data=working, x=left, y=right, ax=axis, scatter_kws={"alpha": 0.55})
    charts["Relationship plot"] = figure
    bins = pd.qcut(working[left], q=min(10, working[left].nunique()), duplicates="drop")
    binned = working.groupby(bins, observed=True).agg({left: "mean", right: "mean"})
    trend, trend_ax = _figure(f"Binned mean {right} by {left}")
    trend_ax.plot(binned[left], binned[right], marker="o")
    trend_ax.set(xlabel=left, ylabel=right)
    charts["Binned means"] = trend
    distribution, dist_ax = _figure(f"Distribution of {left}")
    sns.histplot(working[left], kde=True, ax=dist_ax)
    charts["X distribution"] = distribution
    result_stats = {
        "Analyzed columns": f"{left} and {right}",
        "Rows used": len(working),
        "Rows dropped for missing values": len(data) - len(working),
        "Pearson r": float(pearson.statistic),
        "Pearson p-value": float(pearson.pvalue),
        "Spearman rho": float(spearman.statistic),
        "Spearman p-value": float(spearman.pvalue),
    }
    strength = abs(float(pearson.statistic))
    level = "strong" if strength >= 0.7 else "moderate" if strength >= 0.4 else "weak"
    significance = "statistically significant" if pearson.pvalue < 0.05 else "not statistically significant"
    explanation = (
        f"{left} and {right} have a {level} linear association "
        f"(Pearson r = {pearson.statistic:.3f}, p = {_safe_p(float(pearson.pvalue))}). "
        f"This result is {significance}. Correlation is not causation."
    )
    return RelationshipResult(
        charts=charts, statistics=_stats_table(result_stats),
        explanation=explanation, summary=explanation,
    )


def _categorical_categorical(
    data: pd.DataFrame, left: str, right: str
) -> RelationshipResult:
    """Compare categorical columns with contingency tests and bounded charts."""
    working = data[[left, right]].dropna().copy()
    if len(working) < 5:
        raise ValueError("At least 5 complete rows are needed for this comparison.")
    for column in (left, right):
        counts = working[column].astype(str).value_counts()
        keep = counts.head(20).index
        working[column] = working[column].astype(str).where(
            working[column].astype(str).isin(keep), "Other"
        )
    table = pd.crosstab(working[left], working[right])
    if min(table.shape) < 2:
        raise ValueError("Both columns need at least two observed categories.")
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n * denominator))) if denominator else 0.0
    charts: dict[str, Figure] = {}
    grouped, ax = _figure(f"{right} counts by {left}")
    table.plot(kind="bar", ax=ax)
    ax.tick_params(axis="x", labelrotation=30)
    charts["Grouped counts"] = grouped
    stacked, ax = _figure(f"{right} distribution by {left}")
    (table.div(table.sum(axis=1), axis=0)).plot(kind="bar", stacked=True, ax=ax)
    ax.tick_params(axis="x", labelrotation=30)
    charts["100% stacked"] = stacked
    heatmap, ax = _figure(f"{left} vs {right}")
    sns.heatmap(table, annot=True, fmt="g", cmap="Blues", ax=ax)
    charts["Contingency heatmap"] = heatmap
    percentage, ax = _figure(f"Row percentages: {left} vs {right}")
    sns.heatmap(table.div(table.sum(axis=1), axis=0), annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
    charts["Row percentages"] = percentage
    significant = "statistically significant" if p_value < 0.05 else "not statistically significant"
    explanation = (
        f"{left} and {right} show a Cramer's V association of {cramers_v:.3f}; "
        f"the association is {significant} (chi-square p = {_safe_p(float(p_value))}). "
        "Association does not establish cause; correlation is not causation."
    )
    result_stats = {
        "Analyzed columns": f"{left} and {right}",
        "Rows used": n,
        "Rows dropped for missing values": len(data) - n,
        "Chi-square": float(chi2),
        "p-value": float(p_value),
        "Cramer's V": cramers_v,
    }
    stats_table = _stats_table(result_stats)
    return RelationshipResult(
        charts=charts,
        statistics=stats_table,
        explanation=explanation,
        summary=explanation,
        tables={
            "Counts": table.reset_index(),
            "Row percentages": table.div(table.sum(axis=1), axis=0).reset_index(),
        },
    )


def _datetime_numeric(
    data: pd.DataFrame, date_column: str, numeric_column: str
) -> RelationshipResult:
    """Show a numeric measure over time with a rolling mean."""
    working = pd.DataFrame({
        date_column: pd.to_datetime(data[date_column], errors="coerce", format="mixed"),
        numeric_column: _numeric(data[numeric_column]),
    }).dropna().sort_values(date_column)
    if len(working) < 5:
        raise ValueError("At least 5 complete date-and-number rows are needed.")
    daily = working.groupby(date_column, as_index=False)[numeric_column].mean()
    daily["Rolling mean"] = daily[numeric_column].rolling(
        max(2, min(7, len(daily))), min_periods=1
    ).mean()
    fig, ax = _figure(f"{numeric_column} over time")
    ax.plot(daily[date_column], daily[numeric_column], alpha=0.5, label="Observed")
    ax.plot(daily[date_column], daily["Rolling mean"], label="Rolling mean")
    ax.legend()
    second, second_ax = _figure(f"Distribution of {numeric_column} over time")
    sns.scatterplot(data=working, x=date_column, y=numeric_column, ax=second_ax)
    explanation = (
        f"{numeric_column} is plotted over {date_column}; the rolling line smooths "
        "short-term changes. This pattern is descriptive and does not establish causation."
    )
    return RelationshipResult(
        charts={"Time trend": fig, "Observed values": second},
        statistics=_stats_table({
            "Analyzed columns": f"{date_column} and {numeric_column}",
            "Rows used": len(working),
            "Rows dropped for missing values": len(data) - len(working),
            "Mean": float(working[numeric_column].mean()),
            "Median": float(working[numeric_column].median()),
        }),
        explanation=explanation, summary=explanation,
    )


def _interesting_pairs(dataframe: pd.DataFrame) -> list[tuple[str, str]]:
    """Rank useful continuous/binary, continuous, and binary/category pairs."""
    types = infer_column_types(dataframe)
    numeric = [name for name, kind in types.items() if kind == "numeric_continuous"]
    binary = [name for name, kind in types.items() if kind == "binary"]
    candidates: list[tuple[float, str, str]] = []
    for target in binary:
        target_values = _numeric(dataframe[target])
        if target_values.notna().sum() < 5 or target_values.nunique() != 2:
            continue
        for measure in numeric:
            paired = pd.DataFrame({
                "measure": _numeric(dataframe[measure]), "target": target_values
            }).dropna()
            if len(paired) >= 5 and paired["measure"].nunique() > 1:
                score = abs(float(paired["measure"].corr(paired["target"])))
                if np.isfinite(score):
                    candidates.append((score, measure, target))
        category_columns = [
            name for name, kind in types.items()
            if kind in {"numeric_discrete", "categorical"} and name != target
        ]
        for category in category_columns:
            paired = dataframe[[category, target]].dropna()
            table = pd.crosstab(paired[category], paired[target])
            if len(paired) >= 5 and min(table.shape) > 1:
                chi2 = float(stats.chi2_contingency(table)[0])
                denominator = min(table.shape[0] - 1, table.shape[1] - 1)
                association = float(np.sqrt(chi2 / (len(paired) * denominator)))
                candidates.append((association, category, target))
    for index, left in enumerate(numeric):
        for right in numeric[index + 1:]:
            paired = pd.DataFrame({
                "left": _numeric(dataframe[left]),
                "right": _numeric(dataframe[right]),
            }).dropna()
            if len(paired) >= 5 and paired["left"].nunique() > 1 and paired["right"].nunique() > 1:
                score = abs(float(paired["left"].corr(paired["right"])))
                if np.isfinite(score):
                    candidates.append((score, left, right))

    candidates.sort(reverse=True)
    ranked_pairs = list(dict.fromkeys((left, right) for _, left, right in candidates))
    if ranked_pairs:
        return ranked_pairs[:2]
    usable = [name for name, kind in types.items() if kind not in {"text_id", "datetime"}]
    numeric_usable = [name for name in usable if types[name] in NUMERIC_TYPES]
    if len(numeric_usable) >= 2:
        return [(numeric_usable[0], numeric_usable[1])]
    if len(usable) >= 2:
        return [(usable[0], usable[1])]
    raise ValueError("This dataset needs at least two usable columns for relationship analysis.")


def analyze_relationship(
    dataframe: pd.DataFrame,
    var_a: str | None = None,
    var_b: str | None = None,
    full_analysis: bool = False,
) -> RelationshipResult:
    """Analyze the inferred type pair without mutating the uploaded dataframe."""
    if full_analysis:
        selected_pairs = _interesting_pairs(dataframe)
        pair_results = [
            analyze_relationship(dataframe, left, right)
            for left, right in selected_pairs
        ]
        combined_charts: dict[str, Figure] = {}
        statistics: list[pd.DataFrame] = []
        combined_tables: dict[str, pd.DataFrame] = {}
        for (left, right), result_item in zip(selected_pairs, pair_results):
            for chart_name, figure in result_item.charts.items():
                combined_charts[f"{left} vs {right}: {chart_name}"] = figure
            pair_statistics = result_item.statistics.copy()
            pair_statistics.insert(0, "Relationship", f"{left} vs {right}")
            statistics.append(pair_statistics)
            for table_name, table in result_item.tables.items():
                combined_tables[f"{left} vs {right}: {table_name}"] = table
        explanation = " ".join(
            f"{left} versus {right}: {item.explanation}"
            for (left, right), item in zip(selected_pairs, pair_results)
        )
        combined_statistics = pd.concat(statistics, ignore_index=True)
        result = RelationshipResult(
            charts=combined_charts,
            statistics=combined_statistics,
            explanation=explanation,
            tables=combined_tables,
        )
        result.summary = (
            "Automatically ranked relationship candidates:\n"
            f"{combined_statistics.to_dict(orient='records')}\n"
            "Explain the relationship strength and statistical significance. "
            "Correlation is not causation; do not infer cause."
        )
        return result
    if not var_a or not var_b:
        var_a, var_b = _interesting_pairs(dataframe)[0]
    if var_a == var_b or var_a not in dataframe or var_b not in dataframe:
        raise ValueError("Choose two different columns that exist in the dataset.")
    type_a = infer_column_type(dataframe[var_a], var_a)
    type_b = infer_column_type(dataframe[var_b], var_b)
    if type_a == "datetime" and type_b in NUMERIC_TYPES:
        result = _datetime_numeric(dataframe, var_a, var_b)
    elif type_b == "datetime" and type_a in NUMERIC_TYPES:
        result = _datetime_numeric(dataframe, var_b, var_a)
    elif type_a == type_b == "numeric_continuous":
        result = _numeric_numeric(dataframe, var_a, var_b)
    elif type_a == "numeric_continuous" and type_b in CATEGORY_TYPES:
        result = _numeric_categorical(dataframe, var_a, var_b)
    elif type_b == "numeric_continuous" and type_a in CATEGORY_TYPES:
        result = _numeric_categorical(dataframe, var_b, var_a)
    elif type_a in CATEGORY_TYPES and type_b in CATEGORY_TYPES:
        result = _categorical_categorical(dataframe, var_a, var_b)
    elif type_a in NUMERIC_TYPES and type_b in CATEGORY_TYPES:
        result = _numeric_categorical(dataframe, var_a, var_b)
    elif type_b in NUMERIC_TYPES and type_a in CATEGORY_TYPES:
        result = _numeric_categorical(dataframe, var_b, var_a)
    else:
        raise ValueError(
            f"Cannot compare {var_a} ({type_a}) with {var_b} ({type_b}). "
            "Try a numeric, binary, or categorical column."
        )
    result.summary = (
        f"Columns: {var_a} ({type_a}) and {var_b} ({type_b}).\n"
        f"Statistics: {result.statistics.to_dict(orient='records')}.\n"
        "Explain the relationship, its strength, statistical significance, and "
        "the caveat that correlation is not causation. Do not infer causation."
    )
    return result
