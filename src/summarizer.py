"""Compact, privacy-aware summaries and column resolution for Ask Your Data."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from dataclasses import dataclass

import pandas as pd

from src.type_inference import (
    SYNONYMS,
    infer_column_type,
    infer_column_types,
    match_column_mentions,
)


@dataclass(frozen=True)
class ColumnResolution:
    """Resolved column names and unresolved user terms."""

    columns: tuple[str, ...]
    unresolved: tuple[str, ...] = ()


def build_rich_summary(dataframe: pd.DataFrame) -> str:
    """Summarize dataset metadata and aggregates without exposing raw values."""
    rows, columns = dataframe.shape
    lines = [
        f"Shape: {rows} rows x {columns} columns",
        f"Memory: {dataframe.memory_usage(deep=True).sum()} bytes",
        f"Duplicate rows: {int(dataframe.duplicated().sum())}",
        "Columns:",
    ]
    numeric_columns: list[str] = []
    for column in dataframe.columns:
        series = dataframe[column]
        kind = infer_column_type(series, str(column))
        if kind == "text_id":
            kind = "id_like"
        non_null = series.dropna()
        null_count = int(series.isna().sum())
        null_pct = 100 * null_count / max(1, len(series))
        unique_count = int(non_null.nunique())
        details = [
            f"type={kind}",
            f"dtype={series.dtype}",
            f"nulls={null_count} ({null_pct:.1f}%)",
            f"unique={unique_count}",
        ]
        numeric = pd.to_numeric(series, errors="coerce")
        if kind in {"numeric_continuous", "numeric_discrete", "binary"} and numeric.notna().any():
            numeric_columns.append(str(column))
            details.append(
                "numeric="
                + ", ".join(
                    f"{key}={value:.5g}"
                    for key, value in {
                        "min": numeric.min(),
                        "max": numeric.max(),
                        "mean": numeric.mean(),
                        "median": numeric.median(),
                        "std": numeric.std(),
                    }.items()
                    if pd.notna(value)
                )
            )
        elif kind == "datetime":
            dates = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
            if not dates.empty:
                details.append(f"date_range={dates.min()} to {dates.max()}")
        else:
            top = non_null.value_counts().head(5)
            if not top.empty:
                details.append("top5_counts=" + ", ".join(map(str, top.values)))
        lines.append(f"- {column}: " + "; ".join(details))

    numeric_data = pd.DataFrame({
        column: pd.to_numeric(dataframe[column], errors="coerce")
        for column in numeric_columns
    })
    if len(numeric_columns) >= 2:
        correlations = numeric_data.corr().abs().where(
            ~pd.DataFrame(
                [[i >= j for j in range(len(numeric_columns))]
                 for i in range(len(numeric_columns))],
                index=numeric_columns,
                columns=numeric_columns,
            )
        ).stack().dropna().sort_values(ascending=False).head(10)
        lines.append("Top absolute correlations:")
        for (left, right), value in correlations.items():
            lines.append(f"- {left} vs {right}: |r|={value:.3f}")

    lines.append("Raw rows and category labels are never shared with an AI provider.")
    return "\n".join(lines)


def resolve_columns(request: str, dataframe: pd.DataFrame) -> ColumnResolution:
    """Resolve mentioned columns using the project's synonym and fuzzy matcher."""
    columns = match_column_mentions(
        request, dataframe, limit=min(6, len(dataframe.columns))
    )
    kinds = infer_column_types(dataframe)
    normalized_request = re.sub(r"[^a-z0-9]", "", request.casefold())
    ticket_price_phrase = bool(
        re.search(r"\bticket\s+(?:fare|price)\b", request.casefold())
    )
    columns = [
        column for column in columns
        if kinds.get(column) != "text_id"
        or (
            re.sub(r"[^a-z0-9]", "", column.casefold()) in normalized_request
            and not (ticket_price_phrase and column.casefold() == "ticket")
        )
    ]
    stop_words = {
        "about", "across", "affect", "analyze", "and", "bar", "between",
        "by", "chart", "compare", "correlation", "does", "effect", "for",
        "from", "heatmap", "histogram", "impact", "is", "make", "of", "on",
        "plot", "relationship", "show", "the", "to", "versus", "vs", "what",
    }
    mentions: list[tuple[int, str]] = []
    aliases = {
        str(column): set(SYNONYMS.get(
            re.sub(r"[^a-z0-9]", "", str(column).casefold()), ()
        )) | {str(column)}
        for column in dataframe.columns
    }
    known_terms = {
        token
        for candidates in aliases.values()
        for candidate in candidates
        for token in re.findall(r"[a-z0-9]+", candidate.casefold())
    }
    ticket_fare = bool(re.search(r"\bticket\s+(?:fare|price)\b", request.casefold()))
    for token_match in re.finditer(r"[a-z0-9]+", request.casefold()):
        token = token_match.group()
        if token in stop_words or len(token) < 4:
            continue
        if token in known_terms or (ticket_fare and token == "ticket"):
            continue
        scores = [
            (SequenceMatcher(None, token, alias_token.casefold()).ratio(), column)
            for column, candidates in aliases.items()
            for candidate in candidates
            for alias_token in re.findall(r"[a-z0-9]+", candidate.casefold())
        ]
        if scores:
            best_score = max(score for score, _ in scores)
            if best_score >= 0.80:
                mentions.extend(
                    (token_match.start(), column)
                    for score, column in scores if score == best_score
                )
    for _, column in sorted(mentions):
        if column not in columns:
            columns.append(column)
    columns = [
        column for column in columns
        if kinds.get(column) != "text_id"
        or (
            re.sub(r"[^a-z0-9]", "", column.casefold()) in normalized_request
            and not (ticket_price_phrase and column.casefold() == "ticket")
        )
    ]
    return ColumnResolution(tuple(columns[:6]))
