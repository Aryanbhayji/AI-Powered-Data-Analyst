"""Classify plain-English data requests into analysis intents."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    """A detected request intent and its classification confidence."""

    name: str
    confidence: float
    rule: str


_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("follow_up", ("same but", "now ", "make it a ", "change it to", "instead")),
    ("correlation_heatmap", ("heatmap", "heat map", "correlation matrix")),
    ("full_report", ("analyze this", "analyse this", "full report", "give me insights",
                     "analyze the dataset", "analyse the dataset", "explore this dataset")),
    ("range", ("range of", "min and max", "minimum and maximum", "spread out",
               "how wide", "interquartile", "iqr")),
    ("distribution", ("histogram", "distribution", "show the distribution")),
    ("trend", ("trend", "over time", "by date", "across time", "time series")),
    ("composition", ("composition", "proportion", "share of", "breakdown of",
                     "pie chart", "top categories")),
    ("comparison", ("compare", "comparison", "versus", "vs ", " vs.",
                    "across ", "rate by", "mean by")),
    ("relationship", ("relationship", "related to", "relate to", "affect",
                       "impact of", "effect of", "correlation between")),
    ("summary_stats", ("describe", "summary statistics", "statistics for", "mean of",
                       "average of", "median of", "standard deviation")),
    ("data_cleaning", ("missing values", "missing data", "data quality", "clean the",
                       "duplicates", "duplicate rows")),
)


def classify_intent(request: str) -> Intent:
    """Use ordered keyword rules to classify common analysis requests."""
    text = " ".join(request.casefold().split())
    for name, phrases in _INTENT_RULES:
        if any(phrase in text for phrase in phrases):
            return Intent(name, 0.9, "keyword")
    if re.search(r"\b(?:plot|chart|graph|visuali[sz]e)\b", text):
        return Intent("distribution", 0.65, "generic_plot")
    if text:
        return Intent("unclear", 0.0, "no_rule_match")
    return Intent("unclear", 0.0, "empty_request")
