"""Validate parsed data plans against a dataframe before execution."""

from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from src.command_parser import ALLOWED_ACTIONS, ALLOWED_AGGREGATIONS, ALLOWED_FILTER_OPS
from src.type_inference import infer_column_type, match_column_name

ALLOWED_PLOT_TYPES = {
    "auto", "histogram", "bar", "line", "scatter", "box", "violin",
    "heatmap", "pie", "count", "pair",
}
SUPPORTED_EXAMPLES = (
    "Try 'plot a histogram of a numeric column', "
    "'show rows where a column > 10', or 'average a metric by a category'."
)


class PlanValidationError(ValueError):
    """Raised when a plan is unsafe or incompatible with the uploaded data."""

    def __init__(
        self,
        message: str,
        candidates: tuple[str, ...] = (),
        reference: str | None = None,
    ) -> None:
        """Store a friendly error and optional columns to help resolve ambiguity."""
        super().__init__(message)
        self.candidates = candidates
        self.reference = reference


def _is_numeric_column(dataframe: pd.DataFrame, column: str) -> bool:
    """Check native or string-encoded values for usable numeric data."""
    series = dataframe[column]
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.to_numeric(series, errors="coerce").notna().any()
    )


def _resolve_column(value: Any, dataframe: pd.DataFrame) -> tuple[str, str | None]:
    """Resolve an exact, case-insensitive, or unique fuzzy column reference."""
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError("A required column name is missing.")
    names = [str(column) for column in dataframe.columns]
    matches = [name for name in names if name.casefold() == value.casefold()]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        options = tuple(matches)
        raise PlanValidationError(
            f"The column name {value!r} is ambiguous. Please choose one: "
            + ", ".join(options),
            candidates=options,
            reference=value,
        )

    synonym = match_column_name(value, dataframe)
    if synonym is not None:
        if SequenceMatcher(None, value.casefold(), synonym.casefold()).ratio() >= 0.72:
            return synonym, (
                f"Using '{synonym}' for the similar column name '{value}'."
            )
        return synonym, (
            f"Interpreting '{value}' as the dataset column '{synonym}'."
        )

    ranked = sorted(
        ((SequenceMatcher(None, value.casefold(), name.casefold()).ratio(), name)
         for name in names),
        reverse=True,
    )
    close = [name for score, name in ranked if score >= 0.72]
    if not close:
        available = ", ".join(names[:12]) or "none"
        raise PlanValidationError(
            f"I could not find a column named {value!r}. Available columns: "
            f"{available}. {SUPPORTED_EXAMPLES}"
        )
    best_score = ranked[0][0]
    best = [name for score, name in ranked if score == best_score]
    if len(best) > 1:
        options = tuple(best)
        raise PlanValidationError(
            f"The column name {value!r} could match multiple columns. "
            f"Please choose one: {', '.join(options)}.",
            candidates=options,
            reference=value,
        )
    suggestion = best[0]
    return suggestion, f"Using '{suggestion}' for the similar column name '{value}'."


def _require_numeric(dataframe: pd.DataFrame, column: str, purpose: str) -> None:
    """Require a column to contain numeric or numeric-convertible values."""
    if not _is_numeric_column(dataframe, column):
        raise PlanValidationError(
            f"'{column}' must contain numeric values for {purpose}. "
            f"{SUPPORTED_EXAMPLES}"
        )


def validate_plan(plan: dict[str, Any], dataframe: pd.DataFrame) -> dict[str, Any]:
    """Normalize column references and reject unsupported or incompatible plans."""
    if not isinstance(plan, dict):
        raise PlanValidationError("The interpreted request is not a valid data plan.")
    validated = deepcopy(plan)
    for field in ("title", "explanation"):
        value = validated.get(field, "")
        if not isinstance(value, str):
            raise PlanValidationError(f"The plan's {field} must be plain text.")
        validated[field] = value[:240]
    action = validated.get("action")
    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        raise PlanValidationError(
            "That action is not supported. I can plot, filter, describe, count, "
            "aggregate, correlate, or analyze relationships in the uploaded data."
        )
    if action == "unsupported":
        raise PlanValidationError(
            "I can't perform that request yet. I can plot, filter, describe, count, "
            "aggregate, correlate, or analyze relationships between columns. "
            + SUPPORTED_EXAMPLES
        )
    plot_type = validated.get("plot_type") or "auto"
    repaired_plot_type = False
    if plot_type is not None and (
        not isinstance(plot_type, str) or plot_type not in ALLOWED_PLOT_TYPES
    ):
        plot_type = "auto"
        repaired_plot_type = True
    aggregation = validated.get("aggregation")
    if aggregation is not None and (
        not isinstance(aggregation, str) or aggregation not in ALLOWED_AGGREGATIONS
    ):
        raise PlanValidationError(
            "That aggregation is not supported. Choose mean, sum, median, min, max, or count."
        )

    existing_warnings = validated.get("warnings", [])
    warnings: list[str] = (
        [str(warning) for warning in existing_warnings]
        if isinstance(existing_warnings, list) else []
    )
    if repaired_plot_type:
        warnings.append("I used the closest compatible chart type instead.")
    validated["plot_type"] = plot_type
    for field in ("x", "y", "var_a", "var_b", "group_by", "hue"):
        value = validated.get(field)
        if value is not None:
            try:
                resolved, warning = _resolve_column(value, dataframe)
            except PlanValidationError:
                if action not in {"plot", "relationship", "correlate", "full_analysis"}:
                    raise
                resolved, warning = None, (
                    f"I could not match '{value}', so I selected a compatible column instead."
                )
            validated[field] = resolved
            if warning:
                warnings.append(warning)

    filters = validated.get("filters", [])
    if not isinstance(filters, list):
        raise PlanValidationError("The filter conditions are not in a supported format.")
    normalized_filters = []
    for item in filters:
        if not isinstance(item, dict) or set(item) != {"column", "op", "value"}:
            raise PlanValidationError("Each filter must specify a column, operator, and value.")
        operator = item.get("op")
        if not isinstance(operator, str) or operator not in ALLOWED_FILTER_OPS:
            raise PlanValidationError(
                "That filter operator is not supported. Use >, <, >=, <=, ==, or !=."
            )
        value = item.get("value")
        if isinstance(value, (dict, list, tuple, set)):
            raise PlanValidationError("Filter values must be simple text or numbers.")
        column, warning = _resolve_column(item.get("column"), dataframe)
        if warning:
            warnings.append(warning)
        normalized_filters.append({
            "column": column,
            "op": operator,
            "value": value,
        })
    validated["filters"] = normalized_filters

    top_n = validated.get("top_n")
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1
    ):
        raise PlanValidationError("The requested top-N limit must be a positive whole number.")
    if top_n is not None:
        validated["top_n"] = min(top_n, 1000)
    validated["warnings"] = warnings

    x, y, group_by = validated.get("x"), validated.get("y"), validated.get("group_by")
    if action == "plot":
        if plot_type in {"box", "violin"} and x and not y:
            validated["y"] = x
        if plot_type in {"bar", "pie", "count"} and x is None:
            validated["x"] = group_by or y
    elif action == "aggregate":
        if aggregation is None:
            raise PlanValidationError("Choose an aggregation such as mean, sum, or count.")
        measure = y or (x if x != group_by else None)
        if aggregation != "count":
            if measure is None:
                raise PlanValidationError(
                    f"Name the numeric column to {aggregation}, for example "
                    "'average salary by department'."
                )
            _require_numeric(dataframe, measure, f"{aggregation} aggregation")
            validated["y"] = measure
        if plot_type == "bar" and group_by is None:
            raise PlanValidationError("Name a category to group by for an aggregate bar chart.")
    elif action == "correlate":
        if x is None or y is None:
            raise PlanValidationError("Name two numeric columns to correlate.")
        _require_numeric(dataframe, x, "a correlation")
        _require_numeric(dataframe, y, "a correlation")
    elif action == "relationship":
        variables = [validated.get("var_a"), validated.get("var_b")]
        variables = [column for column in variables if column in dataframe]
        if len(variables) < 2:
            compatible = [
                str(column) for column in dataframe.columns
                if infer_column_type(dataframe[column], str(column)) != "text_id"
            ][:2]
            validated["var_a"] = compatible[0] if compatible else None
            validated["var_b"] = compatible[1] if len(compatible) > 1 else None
            warnings.append("I selected the closest available columns for a plot.")
    elif action == "full_analysis":
        pass
    elif action == "filter" and not normalized_filters:
        raise PlanValidationError(
            "Add a filter condition, such as 'show rows where age > 30'."
        )

    return validated
