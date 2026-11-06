"""Infer semantic column roles and resolve natural-language column mentions."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

COLUMN_TYPES = {
    "numeric_continuous",
    "numeric_discrete",
    "binary",
    "categorical",
    "datetime",
    "text_id",
}

SYNONYMS: dict[str, tuple[str, ...]] = {
    "fare": ("fare", "ticket fare", "ticket price", "price", "cost", "amount paid"),
    "survived": ("survival", "survived", "survivor", "survivors", "survival rate"),
    "sex": ("gender", "sex"),
    "pclass": ("class", "passenger class", "ticket class", "pclass"),
    "age": ("age", "passenger age"),
    "sibsp": ("sibsp", "siblings spouse", "siblings spouses", "siblings", "spouse"),
    "parch": ("parch", "parents children", "parent child", "parents", "children"),
    "embarked": ("embarked", "embarkation", "embarkation port", "port", "ports"),
    "passengerid": ("passenger id", "passenger number", "passenger"),
    "cabin": ("cabin", "deck"),
    "ticket": ("ticket", "ticket number"),
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalized(value: str) -> str:
    """Normalize spaces and punctuation for column-name comparisons."""
    return "".join(_TOKEN_PATTERN.findall(value.casefold()))


def _tokens(value: str) -> set[str]:
    """Split text into lowercase alphanumeric tokens."""
    return set(_TOKEN_PATTERN.findall(value.casefold()))


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse likely date strings without emitting pandas format warnings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    values = series.dropna().astype(str)
    if values.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    date_like = values.str.match(
        r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
    )
    if date_like.mean() < 0.8:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def infer_column_type(series: pd.Series, name: str = "") -> str:
    """Classify a column as numeric, binary, categorical, datetime, or ID-like."""
    values = series.dropna()
    if values.empty:
        return "categorical"
    unique_count = int(values.nunique())
    unique_ratio = unique_count / len(values)
    normalized_name = _normalized(name)
    string_dtype = (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    if (
        normalized_name.endswith(("id", "ticket", "name"))
        and unique_ratio > 0.5
    ):
        return "text_id"
    if string_dtype:
        parsed_dates = _parse_datetime(series)
        if parsed_dates.notna().mean() >= 0.8:
            return "datetime"

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio >= 0.8:
        distinct = int(numeric.nunique())
        if distinct == 2:
            return "binary"
        if unique_ratio > 0.9 and string_dtype:
            return "text_id"
        if unique_ratio > 0.9 and normalized_name.endswith(("id", "number")):
            return "text_id"
        if distinct <= 10:
            return "numeric_discrete"
        return "numeric_continuous"

    if unique_count == 2:
        return "binary"
    if unique_ratio > 0.9 and string_dtype:
        return "text_id"
    return "categorical"


def infer_column_types(dataframe: pd.DataFrame) -> dict[str, str]:
    """Return a semantic type for every dataframe column."""
    return {
        str(column): infer_column_type(dataframe[column], str(column))
        for column in dataframe.columns
    }


def _synonym_candidates(phrase: str, dataframe: pd.DataFrame) -> list[str]:
    """Resolve every distinct column concept named in a request."""
    normalized_phrase = " ".join(_TOKEN_PATTERN.findall(phrase.casefold()))
    candidates: list[tuple[int, int, str]] = []
    for column in dataframe.columns:
        name = str(column)
        canonical = _normalized(name)
        for alias in SYNONYMS.get(canonical, ()):
            alias_tokens = " ".join(_TOKEN_PATTERN.findall(alias.casefold()))
            match = re.search(
                rf"(?<!\w){re.escape(alias_tokens)}(?!\w)", normalized_phrase
            )
            if match:
                candidates.append((match.start(), -len(alias_tokens), name))
    candidates.sort()
    result: list[str] = []
    for _, _, name in candidates:
        if name not in result:
            result.append(name)
    if "fare" in normalized_phrase and "ticket fare" in normalized_phrase:
        result = [name for name in result if _normalized(name) != "ticket"]
    return result


def match_column_mentions(
    text: str,
    dataframe: pd.DataFrame,
    limit: int = 2,
    preferred_types: tuple[str, ...] = (),
) -> list[str]:
    """Find columns mentioned by name, aliases, overlapping tokens, or fuzzy match."""
    types = infer_column_types(dataframe)
    names = [str(column) for column in dataframe.columns]
    normalized_text = _normalized(text)
    exact_matches = [
        name for name in names
        if _normalized(name) and _normalized(name) in normalized_text
    ]

    alias_matches = _synonym_candidates(text, dataframe)
    combined = list(dict.fromkeys(alias_matches + exact_matches))
    if "ticket fare" in " ".join(_TOKEN_PATTERN.findall(text.casefold())):
        combined = [name for name in combined if _normalized(name) != "ticket"]
    if not combined:
        query_tokens = _tokens(text)
        scored: list[tuple[float, str]] = []
        for name in names:
            column_tokens = _tokens(name)
            overlap = len(query_tokens & column_tokens) / max(1, len(column_tokens))
            aliases = SYNONYMS.get(_normalized(name), ())
            alias_overlap = max(
                (len(query_tokens & _tokens(alias)) / max(1, len(_tokens(alias)))
                 for alias in aliases),
                default=0.0,
            )
            score = max(overlap, alias_overlap)
            if score >= 0.5:
                scored.append((score, name))
        if scored:
            best_score = max(score for score, _ in scored)
            combined = [name for score, name in scored if score == best_score]

    if not combined:
        normalized_query = normalized_text
        ranked = sorted(
            (
                SequenceMatcher(None, normalized_query, _normalized(name)).ratio(),
                name,
            )
            for name in names
        )
        if ranked and ranked[-1][0] >= 0.65:
            combined = [ranked[-1][1]]
    if not combined:
        aliases_by_column = {
            name: set(SYNONYMS.get(_normalized(name), ())) | {name}
            for name in names
        }
        fuzzy_mentions: list[tuple[int, str]] = []
        stop_words = {
            "a", "an", "and", "by", "between", "chart", "compare", "correlate",
            "correlation", "does", "effect", "heatmap", "histogram", "how",
            "impact", "is", "of", "on", "or", "plot", "relationship", "relate",
            "related", "show", "the", "to", "versus", "visualize", "vs", "what",
        }
        for match in _TOKEN_PATTERN.finditer(text.casefold()):
            word = match.group()
            if word in stop_words or len(word) < 4:
                continue
            scores = [
                (
                    SequenceMatcher(None, word, alias_word).ratio(),
                    name,
                )
                for name, aliases in aliases_by_column.items()
                for alias in aliases
                for alias_word in _tokens(alias)
            ]
            if scores:
                score = max(item[0] for item in scores)
                if score >= 0.74:
                    best = [name for candidate, name in scores if candidate == score]
                    fuzzy_mentions.extend((match.start(), name) for name in best)
        combined = []
        for _, name in sorted(fuzzy_mentions):
            if name not in combined:
                combined.append(name)

    if preferred_types and len(combined) > 1:
        preferred = [
            name for name in combined
            if types.get(name) in preferred_types
        ]
        if preferred:
            combined = preferred
    return combined[:limit]


def match_column_name(
    mention: str,
    dataframe: pd.DataFrame,
    preferred_types: tuple[str, ...] = (),
) -> str | None:
    """Resolve a single column phrase to its best matching dataframe column."""
    matches = match_column_mentions(
        mention, dataframe, limit=1, preferred_types=preferred_types
    )
    return matches[0] if matches else None
