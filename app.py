"""Streamlit interface for the AI-Data-Analyst application."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import traceback

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
from src.ask_agent import AgentResult, run_analysis
from src.loader import CSVLoadError, load_csv
from src.type_inference import infer_column_types
from src.utils import format_analysis_download, load_project_environment
from src.visualizations import (
    boxplot,
    category_bar,
    correlation_heatmap,
    histogram,
    missingness_bar_chart,
)

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
            {"Column": list(overview["dtypes"]), "Data type": list(overview["dtypes"].values())}
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
        skew = analysis.skewness.rename("Skewness").to_frame()
        st.subheader("Skewness")
        st.dataframe(skew, width="stretch")
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


def _render_visualizations(dataframe: pd.DataFrame) -> None:
    """Let users select a visualization and its target column."""
    numeric_columns = [
        column for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [column for column in dataframe.columns
                           if column not in numeric_columns]
    plot_type = st.selectbox(
        "Visualization",
        ("Correlation heatmap", "Histogram", "Boxplot", "Top categories"),
    )

    if plot_type in {"Histogram", "Boxplot"}:
        if not numeric_columns:
            st.info("This dataset has no numeric columns to plot.")
            return
        column = st.selectbox("Numeric column", numeric_columns)
        figure = histogram(dataframe, column) if plot_type == "Histogram" else boxplot(
            dataframe, column
        )
    elif plot_type == "Top categories":
        if not categorical_columns:
            st.info("This dataset has no categorical columns to plot.")
            return
        column = st.selectbox("Categorical column", categorical_columns)
        unique_count = int(dataframe[column].nunique(dropna=True))
        if unique_count > 50:
            st.warning(
                f"{column} has {unique_count:,} unique values; only the top 20 "
                "categories are shown."
            )
        figure = category_bar(dataframe, column)
    else:
        figure = correlation_heatmap(dataframe)
    _render_figure(figure)


def _get_explanation(summary: str) -> ExplanationResult:
    """Generate an explanation once per dataset summary in the current session."""
    key = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    if st.session_state.get("explanation_key") != key:
        with st.spinner("Preparing a plain-English explanation..."):
            result = explain_summary(summary)
        st.session_state["explanation_key"] = key
        st.session_state["explanation_result"] = result
    return st.session_state["explanation_result"]


def _ask_examples(dataframe: pd.DataFrame) -> list[str]:
    """Create five useful example requests using this dataset's column names."""
    names = [str(column) for column in dataframe.columns]
    kinds = infer_column_types(dataframe)
    numeric = [
        column for column in names
        if kinds.get(column) in {"binary", "numeric_discrete", "numeric_continuous"}
    ]
    numeric.sort(key=lambda column: kinds[column] != "numeric_continuous")
    categorical = [column for column in names if column not in numeric]
    examples: list[str] = []
    if numeric:
        examples.append(f"histogram of {numeric[0]}")
    elif categorical:
        examples.append(f"bar chart of {categorical[0]}")
    if numeric and categorical:
        examples.append(f"relationship between {numeric[0]} and {categorical[0]}")
    elif len(numeric) > 1:
        examples.append(f"scatter plot of {numeric[0]} and {numeric[1]}")
    elif numeric:
        examples.append(f"boxplot of {numeric[0]}")
    if len(numeric) > 1:
        examples.append("heatmap of all features")
    elif categorical:
        examples.append(f"pie chart of {categorical[0]}")
    elif numeric:
        examples.append(f"violin plot of {numeric[0]}")
    if len(numeric) > 1:
        examples.append(f"pairplot of {' and '.join(numeric[:3])}")
    elif numeric and categorical:
        examples.append(f"violin plot of {numeric[0]} by {categorical[0]}")
    elif categorical:
        examples.append(f"count chart of {categorical[0]}")
    elif numeric:
        examples.append(f"bar chart of {numeric[0]}")
    if len(numeric) > 1:
        examples.append(f"line plot of {numeric[0]} and {numeric[1]}")
    elif numeric:
        examples.append(f"pie chart of {numeric[0]}")
    elif categorical:
        examples.append(f"histogram of {categorical[0]}")
    unique_examples = list(dict.fromkeys(examples))
    fallbacks = [
        f"pie chart of {names[0]}" if names else "histogram of all features",
        f"bar chart of {names[-1]}" if names else "bar chart of all features",
        f"boxplot of {names[0]}" if names else "boxplot of all features",
        f"line plot of {names[-1]}" if names else "line plot of all features",
    ]
    for fallback in fallbacks:
        if len(unique_examples) >= 5:
            break
        if fallback not in unique_examples:
            unique_examples.append(fallback)
    return unique_examples[:5]


def _plan_interpretation(plan: dict[str, object], command: str) -> str:
    """Describe the validated structured plan in clear UI text."""
    action = str(plan["action"])
    plot_type = plan.get("plot_type")
    if action == "plot":
        columns = [str(plan[name]) for name in ("x", "y", "group_by") if plan.get(name)]
        details = f" a {plot_type} chart" if plot_type else " a chart"
        if columns:
            details += f" using {', '.join(columns)}"
        return f"I understood: create{details}."
    if action == "aggregate":
        measure = plan.get("y")
        group = plan.get("group_by")
        return (
            f"I understood: calculate {plan.get('aggregation')} of "
            f"{measure or 'rows'}"
            + (f" by {group}" if group else "")
            + "."
        )
    if action == "filter":
        conditions = ", ".join(
            f"{item['column']} {item['op']} {item['value']}"
            for item in plan["filters"]
        )
        return f"I understood: show rows matching {conditions}."
    if action == "describe":
        column = plan.get("x")
        return f"I understood: show descriptive statistics{f' for {column}' if column else ''}."
    if action == "correlate":
        return f"I understood: calculate the correlation between {plan['x']} and {plan['y']}."
    if action == "relationship":
        return (
            f"I understood: analyze the relationship between {plan['var_a']} "
            f"and {plan['var_b']}."
        )
    if action == "full_analysis":
        return "I understood: find the strongest relationships in this dataset."
    if action == "count":
        group = plan.get("group_by")
        return f"I understood: count rows{f' by {group}' if group else ''}."
    return f"I understood: {command}"


def _render_ask_result(
    result: dict[str, object], dataframe: pd.DataFrame
) -> None:
    """Render interpretation and plots before tables, explanation, and follow-ups."""
    del dataframe
    st.markdown(str(result.get("message") or result.get("interpretation", "")))
    figures = result.get("figures")
    if not isinstance(figures, list):
        figures = [result.get("figure")]
    pngs = result.get("pngs", [])
    for index, figure in enumerate(figures):
        if not isinstance(figure, Figure):
            continue
        skewed = result.get("skewed_columns", [])
        if isinstance(skewed, list) and skewed and index == 0:
            log_scale = st.checkbox(
                f"Use log scale for {', '.join(str(column) for column in skewed)}",
                key="ask-log-scale",
            )
            for axis in figure.axes:
                axis.set_yscale("log" if log_scale else "linear")
        _render_figure(figure)
        if isinstance(pngs, list) and index < len(pngs):
            st.download_button(
                f"Download plot {index + 1} as PNG",
                data=pngs[index],
                file_name=f"ask_your_data_plot_{index + 1}.png",
                mime="image/png",
                key=f"ask-chart-download-{index}",
            )
    for warning in result.get("warnings", []):
        st.warning(str(warning))
    tables = result.get("tables", [])
    if isinstance(tables, list):
        for index, table in enumerate(tables):
            if not isinstance(table, pd.DataFrame):
                continue
            st.subheader("Statistics" if index == 0 else f"Additional results {index}")
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                f"Download table {index + 1} as CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"ask_your_data_table_{index + 1}.csv",
                mime="text/csv",
                key=f"ask-table-download-{index}",
            )
    if result.get("explanation_notice"):
        st.caption(str(result["explanation_notice"]))
    if result.get("explanation_text"):
        st.subheader("Plain-English explanation")
        st.write(result["explanation_text"])
    suggestions = result.get("suggestions", [])
    if isinstance(suggestions, list) and suggestions:
        st.subheader("Recommended follow-ups")
        columns = st.columns(min(3, len(suggestions)))
        for index, (column, suggestion) in enumerate(zip(columns, suggestions)):
            if column.button(str(suggestion), key=f"ask-followup-{index}"):
                st.session_state["ask_command"] = str(suggestion)
                st.session_state["ask_run_requested"] = True
                st.rerun()
    report = result.get("markdown_report")
    if isinstance(report, str) and report:
        st.download_button(
            "Download full Markdown report",
            data=report.encode("utf-8"),
            file_name="ask_your_data_report.md",
            mime="text/markdown",
            key="ask-markdown-download",
        )
    if result.get("plan"):
        with st.expander("Show validated analysis plan"):
            st.json(result["plan"])
    if st.session_state.get("ask_debug_mode"):
        with st.expander("Debug details"):
            st.json(result.get("debug", {}))
            if result.get("traceback"):
                st.code(str(result["traceback"]), language="text")


def _process_ask_command(
    command: str,
    dataframe: pd.DataFrame,
    *,
    provider: str,
    model_name: str,
    api_key: str,
) -> None:
    """Run one request through the grounded, validated analysis agent."""
    history = st.session_state.get("ask_history", [])
    previous_request = history[0] if history else None
    previous_result = st.session_state.get("ask_result")
    previous_plan = (
        previous_result.get("plan")
        if isinstance(previous_result, dict)
        and isinstance(previous_result.get("plan"), dict)
        else None
    )
    st.session_state["ask_history"] = ([command] + history)[:10]
    st.session_state["ask_result"] = None
    try:
        with st.spinner("Routing, analyzing, and preparing your results..."):
            result: AgentResult = run_analysis(
                command,
                dataframe,
                previous_request=previous_request,
                previous_plan=previous_plan,
                conversation_context="\n".join(history[:5]),
                provider=provider,
                model_name=model_name,
                api_key=api_key or None,
                use_model=bool(api_key),
            )
        pngs: list[bytes] = []
        for figure in result.figures:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
            pngs.append(buffer.getvalue())
        skewed_columns = []
        for column in result.plan.get("resolved_columns", []):
            if column not in dataframe:
                continue
            numeric = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            if numeric.nunique() > 2 and abs(numeric.skew()) > 1:
                skewed_columns.append(column)
        interpretation = _plan_interpretation(result.plan, command)
        if result.intent.name == "full_report":
            interpretation = "I understood: create a multi-chart report of this dataset."
        elif result.intent.name == "range":
            interpretation = f"I understood: summarize the range of {result.plan.get('x')}."
        elif result.intent.name == "data_cleaning":
            interpretation = "I understood: review missingness, duplicates, and data quality."
        tables = result.tables
        explanation = result.explanation
        report_lines = [
            "# Ask Your Data report",
            "",
            f"Request: {command}",
            f"Intent: {result.intent.name}",
            "",
            "## Explanation",
            explanation.text if explanation else "No explanation was available.",
            "",
            "## Analysis summary",
            result.summary,
        ]
        for index, table in enumerate(tables, start=1):
            report_lines.extend([
                "",
                f"## Table {index}",
                "```text\n" + table.to_string(index=False) + "\n```",
            ])
        st.session_state["ask_result"] = {
            "message": interpretation,
            "interpretation": interpretation,
            "figure": result.figures[0] if result.figures else None,
            "figures": result.figures,
            "pngs": pngs,
            "skewed_columns": skewed_columns,
            "tables": tables,
            "explanation_text": explanation.text if explanation else "",
            "explanation_notice": explanation.notice if explanation else "",
            "summary": result.summary,
            "suggestions": result.suggestions,
            "warnings": result.warnings,
            "plan": result.plan,
            "debug": {
                "intent": result.intent.name,
                "resolved_columns": result.plan.get("resolved_columns", []),
                "planner_mode": result.plan.get("planner_mode", "offline"),
                "retries": max(
                    result.retries,
                    int(result.plan.get("planner_retries", 0)),
                ),
                "raw_model_output": result.plan.get("raw_model_output", ""),
                "validation_result": "validated",
            },
            "markdown_report": "\n".join(report_lines),
            "traceback": "",
        }
    except Exception as exc:
        LOGGER.exception("Ask Your Data agent failed.")
        if st.session_state.get("ask_debug_mode"):
            st.error(str(exc) or "The analysis could not be completed.")
            st.session_state["ask_result"] = {
                "message": "I couldn't complete that analysis.",
                "traceback": traceback.format_exc(),
                "debug": {"request": command},
            }
        else:
            st.error(
                f"I couldn't complete that analysis: {exc}. "
                "Try a simpler request or choose one of the dataset-specific examples."
            )


def _render_ask_your_data(dataframe: pd.DataFrame, dataset_key: str) -> None:
    """Render command examples, execution, downloads, and session history."""
    load_project_environment()
    st.sidebar.toggle("Debug mode", key="ask_debug_mode", value=False)
    provider = st.sidebar.selectbox(
        "AI provider",
        ("openai", "anthropic"),
        index=0 if os.getenv("AI_PROVIDER", "openai").lower() == "openai" else 1,
        key="ask-provider",
    )
    env_model = os.getenv(
        "ANTHROPIC_MODEL" if provider == "anthropic" else "OPENAI_MODEL",
        "claude-3-5-haiku-latest" if provider == "anthropic" else "gpt-4o-mini",
    )
    model_name = st.sidebar.text_input(
        "Model", value=env_model, key="ask-model"
    )
    key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = st.sidebar.text_input(
        "API key (optional)",
        value=os.getenv(key_name, ""),
        type="password",
        key="ask-api-key",
    )
    st.write(
        "Ask about the uploaded data in plain English. AI requests contain only "
        "column metadata and aggregate statistics, never raw rows or category "
        "labels. Only validated plans are run; generated Python is never executed."
    )
    if st.session_state.get("ask_dataset_key") != dataset_key:
        st.session_state["ask_dataset_key"] = dataset_key
        st.session_state["ask_result"] = None
        st.session_state["ask_history"] = []
        st.session_state["ask_ambiguity"] = None
        st.session_state["ask_closest"] = None

    with st.form("ask-command-form"):
        command = st.text_input(
            "What would you like to know?",
            key="ask_command",
            max_chars=500,
            placeholder="For example: plot age vs fare",
        )
        run_clicked = st.form_submit_button("Run", type="primary", key="ask-run")
    queued = st.session_state.pop("ask_run_requested", False)
    if run_clicked or queued:
        _process_ask_command(
            command,
            dataframe,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
        )
    stored_result = st.session_state.get("ask_result")
    if stored_result:
        _render_ask_result(stored_result, dataframe)

    st.subheader("Goal Explorer: questions for this dataset")
    examples = _ask_examples(dataframe)
    example_columns = st.columns(5)
    for index, (column, example) in enumerate(zip(example_columns, examples)):
        if column.button(
            example,
            help=example,
            key=f"ask-example-{index}",
        ):
            st.session_state["ask_command"] = example
            st.session_state["ask_run_requested"] = True
            st.rerun()

    history = st.session_state.get("ask_history", [])
    if history:
        with st.expander("Recent commands"):
            for index, previous in enumerate(history):
                if st.button(
                    previous,
                    key=f"ask-history-{index}",
                    width="content",
                ):
                    st.session_state["ask_command"] = previous
                    st.session_state["ask_run_requested"] = True
                    st.rerun()
    if st.button("Clear conversation", key="ask-clear-conversation"):
        st.session_state["ask_history"] = []
        st.session_state["ask_result"] = None
        st.rerun()


def main() -> None:
    """Render the complete data-analysis application."""
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="AI-Data-Analyst", page_icon="📊", layout="wide")
    st.title("AI-Data-Analyst")
    st.write(
        "Upload a CSV for an automatic profile, interactive charts, and a "
        "beginner-friendly explanation. Only summarized statistics are sent to "
        "an optional AI provider; raw rows are never sent."
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

    overview_tab, missing_tab, stats_tab, charts_tab, ask_tab, ai_tab, download_tab = st.tabs(
        [
            "Overview", "Missing values", "Statistics", "Visualizations",
            "Ask Your Data", "AI explanation", "Download",
        ]
    )
    with overview_tab:
        _render_overview(dataframe, analysis)
    with missing_tab:
        _render_missing_values(analysis)
    with stats_tab:
        _render_statistics(analysis)
    with charts_tab:
        _render_visualizations(dataframe)
    with ask_tab:
        dataset_key = hashlib.sha256(contents).hexdigest()
        _render_ask_your_data(dataframe, dataset_key)
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
