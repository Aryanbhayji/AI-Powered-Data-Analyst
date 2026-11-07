"""Privacy-preserving AI explanations and a deterministic offline fallback."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd

from src.analysis import AnalysisResult
from src.utils import load_project_environment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationResult:
    """An explanation and its AI/fallback status."""

    text: str
    used_ai: bool
    notice: str


def build_analysis_summary(analysis: AnalysisResult) -> str:
    """Build a compact profile summary without including any dataset rows."""
    overview = analysis.overview
    lines = [
        f"Dataset shape: {overview['rows']} rows x {overview['columns']} columns.",
        f"Duplicate rows: {overview['duplicate_rows']}.",
        "Column types: " + (
            ", ".join(f"{name} ({dtype})" for name, dtype in overview["dtypes"].items())
            or "none"
        ) + ".",
    ]

    present_missing = analysis.missing_values[
        analysis.missing_values["Missing Count"] > 0
    ]
    missing = [
        f"{row['Column']} ({row['Missing %']:.1f}%)"
        for _, row in present_missing.iterrows()
    ]
    lines.append("Missingness: " + (", ".join(missing) if missing else "none") + ".")

    numeric_highlights: list[str] = []
    for column, values in analysis.numeric_summary.iterrows():
        mean = values.get("mean")
        median = values.get("50%")
        if pd.notna(mean) and pd.notna(median):
            numeric_highlights.append(
                f"{column}: mean {mean:.3g}, median {median:.3g}"
            )
    lines.append(
        "Numeric highlights: "
        + (", ".join(numeric_highlights[:8]) if numeric_highlights else "none")
        + "."
    )

    correlations: list[tuple[float, str]] = []
    if not analysis.correlations.empty:
        for left_index, left in enumerate(analysis.correlations.columns):
            for right in analysis.correlations.columns[left_index + 1:]:
                value = analysis.correlations.loc[left, right]
                if pd.notna(value):
                    correlations.append(
                        (abs(float(value)), f"{left} and {right} (r={value:.2f})")
                    )
    correlations.sort(reverse=True)
    lines.append(
        "Strongest correlations: "
        + (", ".join(item[1] for item in correlations[:5]) if correlations else "none")
        + "."
    )

    outliers = [
        f"{row['Column']}: {int(row['Outlier Count'])}"
        for _, row in analysis.outliers.iterrows()
        if row["Outlier Count"] > 0
    ]
    lines.append("IQR outliers: " + (", ".join(outliers) if outliers else "none") + ".")
    return "\n".join(lines)


def generate_fallback_explanation(summary: str) -> str:
    """Create a beginner-friendly explanation from the compact summary only."""
    lines = dict(
        line.split(": ", 1) for line in summary.splitlines() if ": " in line
    )
    shape = lines.get("Dataset shape", "Dataset shape was not available.")
    missing = lines.get("Missingness", "not available")
    correlations = lines.get("Strongest correlations", "not available")
    outliers = lines.get("IQR outliers", "not available")
    explanation = [
        f"The dataset contains {shape.removesuffix('.')}",
    ]
    if missing.rstrip(".") == "none":
        explanation.append("No missing values were detected.")
    else:
        explanation.append(f"Some columns have missing values: {missing}")
    if correlations.rstrip(".") != "none":
        explanation.append(f"The strongest relationships are: {correlations}")
    if outliers.rstrip(".") != "none":
        explanation.append(f"Potential unusual values were found: {outliers}")
    explanation.append(
        "Next, check the flagged data-quality issues, confirm unusual values with "
        "domain knowledge, and choose an analysis question before drawing conclusions."
    )
    return " ".join(explanation)


def _get_model_response(
    summary: str, provider: str, api_key: str, model_name: str | None = None
) -> str:
    """Call the configured LangChain chat provider with summary text only."""
    timeout = max(1.0, float(os.getenv("AI_TIMEOUT_SECONDS", "20")))
    prompt = (
        "Explain these computed dataset analysis results to a beginner in simple "
        "English. When correlations are included, describe their direction and "
        "strength and state that correlation is not causation. Mention practical "
        "data-quality caveats and avoid causal claims. Write 3 to 6 short "
        "sentences. The summary contains aggregate statistics and column "
        "metadata only, without raw rows or category labels:\n\n"
        f"{summary}"
    )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = ChatOpenAI(
            model=model_name or model, api_key=api_key, timeout=timeout, max_retries=0
        ).invoke(prompt)
    else:
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        response = ChatAnthropic(
            model=model_name or model, api_key=api_key, timeout=timeout, max_retries=0
        ).invoke(prompt)
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def explain_summary(
    summary: str,
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
) -> ExplanationResult:
    """Use a configured AI provider or explain the same summary offline."""
    load_project_environment()
    selected_provider = (provider or os.getenv("AI_PROVIDER", "openai")).lower()
    key_name = "ANTHROPIC_API_KEY" if selected_provider == "anthropic" else "OPENAI_API_KEY"
    selected_key = api_key or os.getenv(key_name)

    if selected_provider not in {"openai", "anthropic"}:
        notice = f"Unsupported AI provider {selected_provider!r}; showing offline explanation."
    elif not selected_key:
        notice = f"No {key_name} is configured; showing offline explanation."
    else:
        try:
            text = _get_model_response(
                summary, selected_provider, selected_key, model_name
            )
            if text:
                return ExplanationResult(text=text, used_ai=True, notice="")
            notice = "The AI provider returned an empty response; showing offline explanation."
        except Exception:
            LOGGER.warning(
                "AI explanation failed for provider %s; using offline fallback.",
                selected_provider,
            )
            notice = "AI provider request failed; showing offline explanation."

    return ExplanationResult(
        text=generate_fallback_explanation(summary),
        used_ai=False,
        notice=notice,
    )
