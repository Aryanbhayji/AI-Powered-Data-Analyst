"""Grounded Ask Your Data analysis pipeline using validated plans and trusted code."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import chi2_contingency

from src.ai_explainer import ExplanationResult, explain_summary
from src.analysis import analyze_dataframe
from src.code_generator import Planner, generate_plan
from src.intent_router import Intent, classify_intent
from src.plan_executor import PlanExecutionResult, execute_plan
from src.plan_validator import PlanValidationError, validate_plan
from src.summarizer import build_rich_summary, resolve_columns
from src.type_inference import infer_column_types
from src.visualizations import create_command_plot, missingness_bar_chart

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Validated analysis output and diagnostics for the Streamlit interface."""

    intent: Intent
    plan: dict[str, Any]
    figures: list[Figure] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    explanation: ExplanationResult | None = None
    summary: str = ""
    suggestions: list[str] = field(default_factory=list)
    retries: int = 0
    warnings: list[str] = field(default_factory=list)
    execution: PlanExecutionResult | None = None


def chart_quality_issues(
    figure: Figure | None, dataframe: pd.DataFrame, columns: list[str]
) -> list[str]:
    """Check that a result has visible chart structure and usable source values."""
    issues: list[str] = []
    if figure is None:
        return ["No figure was produced."]
    if not figure.axes:
        return ["The figure contains no axes."]
    if not any(axis.get_title() for axis in figure.axes):
        issues.append("The chart has no title.")
    if not any(axis.get_xlabel() or axis.get_ylabel() for axis in figure.axes):
        issues.append("The chart has no axis labels.")
    for column in columns:
        if column in dataframe and dataframe[column].notna().sum() == 0:
            issues.append(f"Column {column!r} contains no usable values.")
    for axis in figure.axes:
        for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
            visible = [label for label in labels if label.get_visible() and label.get_text()]
            if len(visible) > 20:
                issues.append("A categorical axis contains more than 20 labels.")
                break
    return issues


def _column_names(plan: dict[str, Any]) -> list[str]:
    """Return unique real column references from a validated plan."""
    return list(dict.fromkeys(
        str(plan[field])
        for field in ("x", "y", "var_a", "var_b", "group_by", "hue")
        if isinstance(plan.get(field), str)
    ))


def _execute_trusted_plan(
    plan: dict[str, Any], dataframe: pd.DataFrame, request: str
) -> PlanExecutionResult:
    """Run trusted analysis code and fall back to a safe auto-selected chart."""
    try:
        return execute_plan(plan, dataframe)
    except Exception as exc:
        LOGGER.exception("Trusted plan execution failed; trying a safe chart fallback.")
        fallback_plan = {
            "action": "plot",
            "plot_type": "auto",
            "x": plan.get("x"),
            "y": plan.get("y"),
            "title": f"Safe overview for {request[:80]}",
        }
        try:
            figure, notice, _ = create_command_plot(
                dataframe.copy(deep=True), fallback_plan, request
            )
        except Exception as fallback_exc:
            LOGGER.exception("Safe chart fallback failed.")
            raise ValueError(
                "The requested analysis and its safe fallback could not be drawn."
            ) from fallback_exc
        warnings = [f"The requested analysis failed ({exc}); showing a safe chart."]
        if notice:
            warnings.append(notice)
        return PlanExecutionResult(figure=figure, warnings=warnings)


def _suggestions(
    dataframe: pd.DataFrame, columns: tuple[str, ...]
) -> list[str]:
    """Create three relevant follow-up questions from the actual schema."""
    names = list(dataframe.columns)
    types = infer_column_types(dataframe)
    numeric = [name for name, kind in types.items()
               if kind in {"numeric_continuous", "numeric_discrete", "binary"}]
    categorical = [name for name, kind in types.items()
                   if kind in {"binary", "categorical", "numeric_discrete"}]
    relevant = [name for name in columns if name in dataframe]
    output: list[str] = []
    if len(relevant) >= 2:
        output.append(f"Show the relationship between {relevant[0]} and {relevant[1]}")
    if numeric:
        output.append(f"Show the distribution of {numeric[0]}")
    if numeric and categorical:
        output.append(f"Compare {numeric[0]} across {categorical[0]}")
    if len(numeric) > 1:
        output.append(f"Show a correlation heatmap of {', '.join(numeric[:4])}")
    return list(dict.fromkeys(output))[:3]


def _report_plans(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Build four deterministic, dataset-grounded plots for a full report."""
    types = infer_column_types(dataframe)
    numeric = [name for name, kind in types.items()
               if kind == "numeric_continuous"]
    categories = [name for name, kind in types.items()
                  if kind in {"categorical", "numeric_discrete", "binary"}]
    plans: list[dict[str, Any]] = []
    if numeric:
        plans.append({"action": "plot", "plot_type": "histogram", "x": numeric[0]})
    if len(numeric) >= 2:
        plans.append({
            "action": "relationship", "plot_type": "scatter",
            "x": numeric[0], "y": numeric[1], "var_a": numeric[0], "var_b": numeric[1],
        })
    if numeric and categories:
        plans.append({
            "action": "relationship", "plot_type": "box", "x": numeric[0],
            "y": categories[0], "var_a": numeric[0], "var_b": categories[0],
            "group_by": categories[0],
        })
    if len(numeric) >= 2:
        plans.append({"action": "plot", "plot_type": "heatmap"})
    elif categories:
        plans.append({"action": "plot", "plot_type": "bar", "x": categories[0]})
    return plans[:6]


def _range_table(dataframe: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame | None:
    """Calculate range and quartile values for the numeric column in a range request."""
    column = plan.get("x")
    if not isinstance(column, str) or column not in dataframe:
        return None
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return None
    quantiles = values.quantile([0.25, 0.5, 0.75])
    return pd.DataFrame({
        "Statistic": ["Min", "Max", "Range", "Q1", "Median", "Q3", "IQR"],
        "Value": [
            values.min(), values.max(), values.max() - values.min(),
            quantiles.loc[0.25], quantiles.loc[0.5], quantiles.loc[0.75],
            quantiles.loc[0.75] - quantiles.loc[0.25],
        ],
    })


def _relationship_metric(
    dataframe: pd.DataFrame, plan: dict[str, Any]
) -> str | None:
    """Return one compact relationship statistic for a chart subtitle/title."""
    left = plan.get("var_a") or plan.get("x")
    right = plan.get("var_b") or plan.get("y")
    if not isinstance(left, str) or not isinstance(right, str):
        return None
    if left not in dataframe or right not in dataframe or left == right:
        return None
    types = infer_column_types(dataframe)
    if types.get(left) == "numeric_continuous" and types.get(right) == "numeric_continuous":
        pair = dataframe[[left, right]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(pair) >= 3 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
            correlation = float(pair[left].corr(pair[right]))
            return f"Pearson r={correlation:.2f}"
    if types.get(left) in {"categorical", "binary", "numeric_discrete"} and (
        types.get(right) in {"categorical", "binary", "numeric_discrete"}
    ):
        table = pd.crosstab(dataframe[left], dataframe[right])
        if table.shape[0] > 1 and table.shape[1] > 1:
            statistic, p_value, _, _ = chi2_contingency(table)
            return f"Chi-square={statistic:.2f}, p={p_value:.3g}"
    continuous = next(
        (column for column in (left, right)
         if types.get(column) == "numeric_continuous"),
        None,
    )
    group = right if continuous == left else left if continuous == right else None
    if continuous and group:
        values = pd.to_numeric(dataframe[continuous], errors="coerce")
        medians = values.groupby(dataframe[group]).median().dropna()
        if len(medians) >= 2:
            return f"median range={medians.max() - medians.min():.3g}"
    return None


def _make_explanation(
    summary: str,
    provider: str | None,
    api_key: str | None,
    model_name: str | None,
) -> ExplanationResult:
    """Explain only the compact profile and computed statistics."""
    return explain_summary(
        summary, provider=provider, api_key=api_key, model_name=model_name
    )


def _offline_result_explanation(
    dataframe: pd.DataFrame,
    intent: Intent,
    plan: dict[str, Any],
) -> str:
    """Explain the requested result with data-derived values and cautious wording."""
    columns = [column for column in _column_names(plan) if column in dataframe]
    types = infer_column_types(dataframe)
    numeric = [
        column for column in columns
        if types.get(column) == "numeric_continuous"
    ]
    categorical = [
        column for column in columns
        if types.get(column) in {"binary", "numeric_discrete", "categorical"}
    ]
    binary_columns = [
        column for column in categorical if types.get(column) == "binary"
    ]
    if binary_columns and len(categorical) >= 2:
        outcome = next(
            (column for column in binary_columns
             if column.casefold().replace("_", "").replace(" ", "") in {
                 "survived", "survival"
             }),
            binary_columns[0],
        )
        grouping = next(
            (column for column in categorical if column != outcome), None
        )
        if grouping:
            outcome_values = pd.to_numeric(
                dataframe[outcome], errors="coerce"
            )
            valid = outcome_values.isin([0, 1])
            rates = outcome_values[valid].groupby(
                dataframe.loc[valid, grouping], observed=True
            ).mean().sort_values()
            if len(rates) >= 2:
                return (
                    f"The {outcome} rate varies across {grouping}: "
                    f"{rates.index[0]} has {rates.iloc[0]:.1%} and "
                    f"{rates.index[-1]} has {rates.iloc[-1]:.1%}. The comparison "
                    f"uses {int(valid.sum()):,} rows with known {outcome} values. "
                    "This association does not establish cause and effect."
                )
    if intent.name == "range" and numeric:
        values = pd.to_numeric(dataframe[numeric[0]], errors="coerce").dropna()
        if not values.empty:
            return (
                f"{numeric[0]} ranges from {values.min():.3g} to {values.max():.3g}, "
                f"a spread of {values.max() - values.min():.3g}. The median is "
                f"{values.median():.3g} and the interquartile range is "
                f"{values.quantile(.75) - values.quantile(.25):.3g}. "
                "The chart and table use non-missing values only."
            )
    if intent.name == "trend":
        x_column, y_column = plan.get("x"), plan.get("y")
        if isinstance(x_column, str) and isinstance(y_column, str):
            working = dataframe[[x_column, y_column]].copy()
            working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
            dates = pd.to_datetime(working[x_column], errors="coerce", format="mixed")
            working = working.assign(_date=dates).dropna(subset=["_date", y_column])
            if not working.empty:
                ordered = working.sort_values("_date")
                first, last = ordered.iloc[0], ordered.iloc[-1]
                return (
                    f"{y_column} changed from {first[y_column]:.3g} on "
                    f"{first['_date'].date()} to {last[y_column]:.3g} on "
                    f"{last['_date'].date()} across {len(ordered):,} dated rows. "
                    "The chart shows an observed trend, not proof of what caused it. "
                    "Check seasonality and missing dates before forecasting."
                )
    if intent.name == "full_report":
        numeric_frame = dataframe.select_dtypes(include="number")
        correlations = numeric_frame.corr().abs().where(
            ~pd.DataFrame(
                [[i >= j for j in range(len(numeric_frame.columns))]
                 for i in range(len(numeric_frame.columns))],
                index=numeric_frame.columns, columns=numeric_frame.columns,
            )
        ).stack().dropna().sort_values(ascending=False)
        finding = (
            f"The strongest numeric association is {correlations.index[0][0]} "
            f"with {correlations.index[0][1]} (|r|={correlations.iloc[0]:.2f})."
            if not correlations.empty else "No numeric column pair had enough values for correlation."
        )
        missing = int(dataframe.isna().sum().sum())
        return (
            f"The report includes {len(dataframe):,} rows and "
            f"{dataframe.shape[1]:,} columns. {finding} There are {missing:,} "
            f"missing cells and {int(dataframe.duplicated().sum()):,} duplicate "
            "rows to review. Correlation is not causation; check data quality and "
            "domain context before drawing conclusions."
        )
    if intent.name == "data_cleaning":
        missing = int(dataframe.isna().sum().sum())
        duplicate_count = int(dataframe.duplicated().sum())
        return (
            f"The dataset contains {missing:,} missing cell(s) and "
            f"{duplicate_count:,} duplicate row(s). Review missingness by column "
            "before choosing a cleaning strategy. Duplicate detection compares "
            "complete rows; correlation is not causation."
        )
    if len(numeric) >= 2:
        paired = dataframe[numeric[:2]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(paired) > 1 and paired.iloc[:, 0].nunique() > 1 and paired.iloc[:, 1].nunique() > 1:
            correlation = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
            strength = "strong" if abs(correlation) >= .7 else "moderate" if abs(correlation) >= .3 else "weak"
            direction = "positive" if correlation >= 0 else "negative"
            return (
                f"{numeric[0]} and {numeric[1]} have a {strength} {direction} "
                f"linear association (Pearson r={correlation:.2f}, using "
                f"{len(paired):,} complete rows). The plot shows association only. "
                "Correlation does not prove causation."
            )
    if numeric and categorical:
        working = pd.DataFrame({
            "value": pd.to_numeric(dataframe[numeric[0]], errors="coerce"),
            "group": dataframe[categorical[0]],
        }).dropna()
        if types.get(categorical[0]) == "binary":
            binary_values = pd.to_numeric(working["group"], errors="coerce")
            if binary_values.notna().all() and binary_values.isin([0, 1]).all():
                bins = pd.qcut(working["value"], q=4, duplicates="drop")
                rates = working.assign(
                    _bin=bins, _outcome=binary_values
                ).groupby("_bin", observed=True)["_outcome"].mean()
                if len(rates) >= 2:
                    return (
                        f"The {categorical[0]} rate is {rates.iloc[0]:.1%} in the "
                        f"lowest {numeric[0]} bin and {rates.iloc[-1]:.1%} in the "
                        f"highest bin. The comparison uses {len(working):,} "
                        "complete rows; this association does not establish a "
                        "causal effect. Correlation is not causation."
                    )
        medians = working.groupby("group", observed=True)["value"].median().sort_values()
        if not medians.empty:
            low_group, high_group = medians.index[0], medians.index[-1]
            return (
                f"Median {numeric[0]} differs across {categorical[0]} groups: "
                f"{low_group} has {medians.iloc[0]:.3g}, while {high_group} has "
                f"{medians.iloc[-1]:.3g}. The chart uses {len(working):,} complete "
                "rows and helps compare distributions. This pattern is not proof "
                "that group membership causes the difference."
            )
    if len(categorical) >= 2:
        working = dataframe[categorical[:2]].dropna()
        counts = pd.crosstab(working.iloc[:, 0], working.iloc[:, 1])
        if not counts.empty:
            return (
                f"The chart compares {categorical[0]} and {categorical[1]} across "
                f"{len(working):,} complete rows. The most frequent combination is "
                f"{counts.stack().idxmax()!r}. This is a descriptive association, "
                "not evidence of causation."
            )
    if numeric:
        values = pd.to_numeric(dataframe[numeric[0]], errors="coerce").dropna()
        if not values.empty:
            return (
                f"{numeric[0]} has a mean of {values.mean():.3g} and a median of "
                f"{values.median():.3g} across {len(values):,} non-missing rows. "
                "Compare the mean and median to assess skew or unusual values. "
                "These summaries do not establish cause and effect."
            )
    return (
        f"The {intent.name.replace('_', ' ')} result summarizes the uploaded "
        "dataset using available non-missing values. Review the plot and table "
        "together before drawing conclusions. Correlation is not causation."
    )


def run_analysis(
    request: str,
    dataframe: pd.DataFrame,
    *,
    previous_request: str | None = None,
    previous_plan: dict[str, Any] | None = None,
    conversation_context: str = "",
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    planner: Planner | None = None,
    use_model: bool = False,
) -> AgentResult:
    """Route, summarize, validate, execute, quality-check, and explain one request."""
    intent = classify_intent(request)
    plan = generate_plan(
        request,
        dataframe,
        intent,
        previous_request=previous_request,
        previous_plan=previous_plan,
        context=conversation_context,
        planner=planner,
        use_model=use_model,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
    )
    planner_warnings: list[str] = []
    try:
        validated = validate_plan(plan, dataframe)
    except PlanValidationError:
        if plan.get("planner_mode") != "model":
            raise
        planner_warnings.append(
            "The AI plan did not match this dataset; I switched to a trusted offline plan."
        )
        plan = generate_plan(
            request,
            dataframe,
            intent,
            previous_request=previous_request,
            previous_plan=previous_plan,
            context=conversation_context,
        )
        validated = validate_plan(plan, dataframe)
    if validated["action"] == "unsupported":
        raise ValueError("I couldn't map that request to a supported analysis.")

    if intent.name == "full_report":
        figures = []
        result: PlanExecutionResult | None = None
        for report_plan in _report_plans(dataframe):
            checked = validate_plan(report_plan, dataframe)
            report_result = _execute_trusted_plan(checked, dataframe, request)
            if report_result.figure is not None:
                figures.append(report_result.figure)
                if result is None:
                    result = report_result
        if figures:
            assert result is not None
            result.figure = figures[0]
        else:
            result = _execute_trusted_plan(validated, dataframe, request)
    else:
        result = _execute_trusted_plan(validated, dataframe, request)
        figures = [result.figure] if result.figure is not None else []
    if not figures and intent.name == "data_cleaning":
        result.figure = missingness_bar_chart(
            analyze_dataframe(dataframe).missing_values
        )
        figures = [result.figure]
    elif not figures and dataframe.shape[1]:
        fallback_plan = {
            "action": "plot",
            "plot_type": "auto",
            "x": validated.get("x"),
            "y": validated.get("y"),
            "title": f"Overview for {intent.name.replace('_', ' ')}",
        }
        fallback_figure, fallback_notice, _ = create_command_plot(
            dataframe.copy(deep=True), fallback_plan, request
        )
        result.figure = fallback_figure
        figures = [fallback_figure]
        if fallback_notice:
            result.warnings.append(fallback_notice)
    if result is None:
        raise ValueError("No compatible analysis could be built for this dataset.")
    if not figures and result.figure is not None:
        figures = [result.figure]

    range_table = _range_table(dataframe, validated) if intent.name == "range" else None
    if range_table is not None and figures:
        values = dict(zip(range_table["Statistic"], range_table["Value"]))
        annotation = (
            f"min={values['Min']:.3g}  max={values['Max']:.3g}\n"
            f"Q1={values['Q1']:.3g}  median={values['Median']:.3g}  "
            f"Q3={values['Q3']:.3g}  IQR={values['IQR']:.3g}"
        )
        figures[0].axes[0].text(
            0.98, 0.98, annotation, ha="right", va="top",
            transform=figures[0].axes[0].transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    retries = 0
    warnings = planner_warnings + list(result.warnings)
    relationship_metric = None
    if intent.name in {"relationship", "comparison"} and figures:
        relationship_metric = _relationship_metric(dataframe, validated)
        if relationship_metric:
            axis = figures[0].axes[0]
            axis.set_title(f"{axis.get_title()} ({relationship_metric})")
    for index, figure in enumerate(list(figures)):
        issues = chart_quality_issues(
            figure, dataframe, _column_names(validated) if index == 0 else []
        )
        if not issues:
            continue
        retries = min(3, retries + 1)
        warnings.extend(issues)
        fallback_plan = {
            **validated,
            "action": "plot",
            "plot_type": "auto",
        }
        try:
            fallback, notice, _ = create_command_plot(
                dataframe.copy(deep=True), fallback_plan, request
            )
            if not chart_quality_issues(fallback, dataframe, []):
                plt.close(figure)
                figures[index] = fallback
                if index == 0:
                    result.figure = fallback
                if notice:
                    warnings.append(notice)
            else:
                plt.close(fallback)
        except Exception:
            LOGGER.exception("Trusted chart repair failed.")
    if intent.name == "full_report" and len(figures) < 4:
        warnings.append("This dataset supports fewer than four distinct report charts.")

    summary = build_rich_summary(dataframe)
    computed = result.summary or result.explanation or result.text or ""
    if relationship_metric:
        computed += f"\nChart statistic: {relationship_metric}"
    if result.statistics is not None:
        computed += "\nComputed statistics:\n" + result.statistics.to_string(index=False)
    if result.table is not None and intent.name == "range":
        computed += "\nResult table:\n" + result.table.to_string(index=False)
    explanation = _make_explanation(
        f"{summary}\nRequest: {request}\nComputed result:\n{computed}",
        provider,
        api_key,
        model_name,
    )
    if not explanation.used_ai:
        explanation = ExplanationResult(
            text=_offline_result_explanation(dataframe, intent, validated),
            used_ai=False,
            notice=explanation.notice,
        )

    tables = [table for table in (result.statistics, result.table) if table is not None]
    if range_table is not None:
        tables.append(range_table)
    return AgentResult(
        intent=intent,
        plan=validated,
        figures=figures,
        tables=tables,
        explanation=explanation,
        summary=summary,
        suggestions=_suggestions(dataframe, tuple(_column_names(validated))),
        retries=retries,
        warnings=warnings,
        execution=result,
    )
