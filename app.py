"""Streamlit interface for the AI-Data-Analyst application."""

from __future__ import annotations

import hashlib
import io
import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.ai_explainer import (
    ExplanationResult,
    build_analysis_summary,
    explain_summary,
)
from src.analysis import AnalysisResult, analyze_dataframe
from src.auto_visualizer import VisualizationReport, build_visualization_report
from src.loader import CSVLoadError, load_csv
from src.utils import format_analysis_download
from src.visualizations import missingness_bar_chart

LOGGER = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _load_cached(contents: bytes, filename: str) -> pd.DataFrame:
    """Cache CSV parsing by file content and name."""
    return load_csv(contents, filename=filename)


@st.cache_data(show_spinner=False)
def _analyze_cached(dataframe: pd.DataFrame) -> AnalysisResult:
    """Cache reusable analysis results for the current dataframe."""
    return analyze_dataframe(dataframe)


def _render_figure(figure: Figure) -> None:
    """Render a Matplotlib figure and release it even if rendering fails."""
    try:
        st.pyplot(figure, clear_figure=False)
    finally:
        plt.close(figure)


def _render_overview(dataframe: pd.DataFrame, analysis: AnalysisResult) -> None:
    """Render the dataset overview and preview."""
    overview = analysis.overview
    first, second, third, fourth = st.columns(4)
    first.metric("Rows", f"{overview['rows']:,}")
    second.metric("Columns", f"{overview['columns']:,}")
    third.metric("Memory", f"{overview['memory_bytes'] / 1024:,.1f} KB")
    fourth.metric("Duplicate rows", f"{overview['duplicate_rows']:,}")

    if overview["rows"] == 0:
        st.warning("This CSV has a header but no data rows.")
    if dataframe.attrs.get("malformed_rows_skipped", 0):
        st.warning(
            f"Skipped {dataframe.attrs['malformed_rows_skipped']} malformed row(s) "
            "while loading the CSV."
        )
    st.subheader("Column types")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": list(overview["dtypes"]),
                "Data type": list(overview["dtypes"].values()),
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("First 10 rows")
    st.dataframe(dataframe.head(10), width="stretch", hide_index=True)


def _render_missing_values(analysis: AnalysisResult) -> None:
    """Render missing-value counts, percentages, and a chart."""
    missing = analysis.missing_values
    st.dataframe(missing, width="stretch", hide_index=True)
    high_missing = missing[missing["Missing %"] > 30]
    if not high_missing.empty:
        names = ", ".join(high_missing["Column"].astype(str))
        st.warning(f"More than 30% of values are missing in: {names}.")
    _render_figure(missingness_bar_chart(missing))


def _render_statistics(analysis: AnalysisResult) -> None:
    """Render numeric and categorical statistics."""
    if analysis.numeric_summary.empty:
        st.info("No numeric columns were detected.")
    else:
        st.subheader("Numeric summary")
        st.dataframe(analysis.numeric_summary, width="stretch")
        st.subheader("Skewness")
        st.dataframe(
            analysis.skewness.rename("Skewness").to_frame(), width="stretch"
        )
        st.subheader("IQR-based outlier counts")
        st.dataframe(analysis.outliers, width="stretch", hide_index=True)

    if not analysis.categorical_summary:
        st.info("No categorical columns were detected.")
    else:
        st.subheader("Categorical columns")
        for column, details in analysis.categorical_summary.items():
            with st.expander(f"{column}: {details['unique_count']:,} unique values"):
                top_values = details["top_values"].rename_axis("Value").rename("Count")
                st.dataframe(top_values, width="stretch")


def _visualization_payload(
    dataframe: pd.DataFrame, dataset_key: str
) -> dict[str, object]:
    """Build and retain automatic chart images for the current uploaded file."""
    if st.session_state.get("visualization_dataset_key") == dataset_key:
        cached = st.session_state.get("visualization_payload")
        if isinstance(cached, dict):
            return cached

    report: VisualizationReport = build_visualization_report(dataframe)
    charts: list[dict[str, object]] = []
    for index, card in enumerate(report.charts):
        image = io.BytesIO()
        try:
            card.figure.savefig(image, format="png", dpi=150, bbox_inches="tight")
        finally:
            plt.close(card.figure)
        charts.append({
            "title": card.title,
            "description": card.description,
            "image": image.getvalue(),
            "index": index,
        })
    payload: dict[str, object] = {
        "insights": report.insights,
        "notices": report.notices,
        "charts": charts,
    }
    st.session_state["visualization_dataset_key"] = dataset_key
    st.session_state["visualization_payload"] = payload
    return payload


def _render_visualizations(dataframe: pd.DataFrame, dataset_key: str) -> None:
    """Render automatic findings and analyst-selected charts."""
    st.subheader("What stands out")
    try:
        with st.spinner("Scanning columns and selecting useful charts..."):
            payload = _visualization_payload(dataframe, dataset_key)
    except Exception:
        LOGGER.exception("Automatic visualization report failed.")
        st.error("Automatic visual analysis failed for this file. Try another CSV.")
        return

    for notice in payload.get("notices", []):
        st.info(str(notice))
    for insight in payload.get("insights", []):
        st.markdown(f"- {insight}")

    charts = payload.get("charts", [])
    if not isinstance(charts, list) or not charts:
        st.info(
            "There are no informative charts for these columns. Try a dataset "
            "with non-constant numeric values, categories, or dates."
        )
        return

    st.subheader("Recommended visualizations")
    for start in range(0, len(charts), 2):
        columns = st.columns(2)
        for column, chart in zip(columns, charts[start:start + 2]):
            if not isinstance(chart, dict):
                continue
            with column:
                st.markdown(f"**{chart['title']}**")
                st.caption(str(chart["description"]))
                image = chart["image"]
                st.image(image, use_container_width=True)
                st.download_button(
                    "Download chart",
                    data=image,
                    file_name=f"data_story_chart_{int(chart['index']) + 1}.png",
                    mime="image/png",
                    key=f"data-story-download-{dataset_key[:10]}-{chart['index']}",
                )


def _get_explanation(summary: str) -> ExplanationResult:
    """Generate an explanation once per dataset summary in the current session."""
    key = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    if st.session_state.get("explanation_key") != key:
        with st.spinner("Preparing a plain-English explanation..."):
            result = explain_summary(summary)
        st.session_state["explanation_key"] = key
        st.session_state["explanation_result"] = result
    return st.session_state["explanation_result"]


def main() -> None:
    """Render the complete data-analysis application."""
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="AI-Data-Analyst", page_icon="📊", layout="wide")
    st.title("AI-Data-Analyst")
    st.write(
        "Upload a CSV for an automatic profile, analyst-selected visualizations, "
        "and a beginner-friendly explanation. Only summarized statistics are "
        "sent to an optional AI provider; raw rows are never sent."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file (maximum 50 MB)",
        type=["csv"],
        help="CSV files are decoded using common UTF-8 and Western encodings.",
    )
    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    contents = uploaded_file.getvalue()
    try:
        dataframe = _load_cached(contents, uploaded_file.name)
    except CSVLoadError as exc:
        st.error(str(exc))
        return
    except Exception:
        LOGGER.exception("Unexpected error while loading uploaded CSV.")
        st.error("The CSV could not be loaded. Check its format and try again.")
        return

    try:
        analysis = _analyze_cached(dataframe)
    except Exception:
        LOGGER.exception("Unexpected error while analyzing uploaded CSV.")
        st.error("The CSV loaded, but its analysis could not be completed.")
        return

    overview_tab, missing_tab, stats_tab, charts_tab, ai_tab, download_tab = st.tabs(
        ["Overview", "Missing values", "Statistics", "Data Story", "AI explanation", "Download"]
    )
    with overview_tab:
        _render_overview(dataframe, analysis)
    with missing_tab:
        _render_missing_values(analysis)
    with stats_tab:
        _render_statistics(analysis)
    with charts_tab:
        dataset_key = hashlib.sha256(contents).hexdigest()
        _render_visualizations(dataframe, dataset_key)

    summary = build_analysis_summary(analysis)
    with ai_tab:
        explanation = _get_explanation(summary)
        if explanation.notice:
            st.warning(explanation.notice)
        if not explanation.used_ai:
            st.caption("Offline rule-based explanation")
        st.write(explanation.text)
    with download_tab:
        st.write("Export the profile and explanation as a text or Markdown report.")
        file_format = st.selectbox("Report format", ("txt", "md"))
        content, mime_type, extension = format_analysis_download(
            summary, explanation.text, file_format
        )
        st.download_button(
            "Download analysis report",
            data=content,
            file_name=f"analysis_report.{extension}",
            mime=mime_type,
        )


if __name__ == "__main__":
    main()
