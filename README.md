# AI-Powered-CSV-Data-Analyst

AI-Data-Analyst is a local Streamlit application that profiles CSV files,
visualizes common patterns, and explains the results in plain English. It can
run entirely offline; an OpenAI or Anthropic key is optional.

## Features

- Upload `.csv` files up to 50 MB with delimiter and common encoding detection.
- Review dataset shape, memory use, duplicates, types, and a 10-row preview.
- Find missing-value counts and percentages, including columns above 30%.
- Explore numeric descriptions, categorical counts, skewness, and IQR outliers.
- Build histograms, category bars, correlation heatmaps, and boxplots.
- Ask Your Data questions in plain English to create charts, filter rows,
  describe columns, count categories, aggregate values, compare ranges and
  trends, or analyze relationships. Requests are routed by intent, grounded in
  a compact data profile, and displayed as large Matplotlib charts with
  explanations. Full reports include multiple plots. Follow-up suggestions,
  the last ten requests, CSV/PNG/Markdown downloads, and optional AI planning
  are available in the app.
- Generate an AI explanation from a compact profile summary. Dataset rows
  and category labels are never sent to an AI provider.
- Use a rule-based explanation when no key is configured or a provider fails.
- Download a TXT or Markdown analysis report.

### Ask Your Data examples

- `plot a histogram of age`
- `show correlation between price and rating`
- `average salary by department as a bar chart`
- `top 10 cities by count`
- `boxplot of income by gender`
- `show rows where age > 30`
- `Relationship between ticket fare and survival`
- `Does age affect survival?`
- `Survival rate by passenger class`
- `Compare fare across embarkation ports`
- `Is gender related to survival?`
- `Correlation between age and fare`
- `Impact of family size (SibSp) on survival`
- `Analyze this dataset`

### Ask Your Data architecture

Ask Your Data uses seven stages:

1. **Intent router** classifies common requests with keyword rules first.
2. **Data summary and column resolver** report inferred types, nulls,
   descriptive statistics, category counts, dates, and correlations, then map
   synonyms and misspellings to real columns. Raw rows and category labels are
   never sent to an AI provider.
3. **Plan generator** returns a constrained JSON analysis plan. It does not
   generate executable Python.
4. **Validation and execution** validate columns and chart parameters, then
   call trusted pandas/Matplotlib functions. Model-generated code is never
   passed to `eval`, `exec`, or a subprocess.
5. **Repair** retries a chart using a compatible trusted chart choice if the
   first result fails validation or has no usable chart.
6. **Chart quality check** verifies that the output has a figure, title,
   labels, and bounded categorical axes.
7. **Explanation and Goal Explorer** explain computed summaries and offer
   dataset-specific follow-up questions.

This constrained-plan design deliberately replaces arbitrary generated-code
execution. An AST filter and restricted Python builtins are not a dependable
security sandbox, and a subprocess alone does not isolate access to files,
network, or the host. The application therefore executes only its own
validated analysis functions. This reduces risk but is not a formal security
boundary; the app is intended for local use with the user's own data. When an
AI provider is configured, only column metadata and aggregate statistics are
sent. Raw rows and category labels are excluded.

## Screenshots

Add screenshots of the Overview, Statistics, and AI explanation tabs here.

## Setup

Python 3.10 or newer is required.

1. Create and activate a virtual environment:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the pinned dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Optional: copy `.env.example` to `.env` and add a provider key as described
   below.

4. Start the app from the project directory:

   ```powershell
   streamlit run app.py
   ```

The project includes sample files in `sample_data/`; upload either one in the
app to try the workflow.

## Optional AI provider keys

Copy `.env.example` to `.env`. Set `AI_PROVIDER` to `openai` or `anthropic` and
provide the matching `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The model can be
overridden with `OPENAI_MODEL` or `ANTHROPIC_MODEL`. `AI_TIMEOUT_SECONDS`
controls the request timeout (20 seconds by default). Keep `.env` private; it
is ignored by Git. Without a key, the app provides an offline explanation
without requiring an internet connection.

## Tests

```powershell
python -m pytest
```

Tests cover CSV loading, encoding fallback, missing values, mixed-type
statistics, outliers, summary privacy, AI fallback behavior, visualizations,
report formatting, natural-language intent routing, column matching, plan
validation, chart quality/repair, and AppTest workflows. Unit tests do not
require a real API key or network access.

## Visualization evaluation

The local evaluation covers at least 40 requests across the Titanic and messy
sales fixtures:

```powershell
python evaluation\run_eval.py
```

If `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is configured, the script runs with
the selected provider and writes per-request outcomes to
`evaluation\results.csv`. To force the deterministic offline path:

```powershell
python evaluation\run_eval.py --offline
```

The evaluation reports the percentage of requests that fail to produce a
titled, labeled figure. Offline results measure deterministic coverage, not
LLM accuracy. Live provider results vary by model and dataset; no particular
error-rate target is guaranteed.

## Project structure

```text
AI-Data-Analyst/
├── app.py
├── src/
│   ├── __init__.py
│   ├── ai_explainer.py
│   ├── analysis.py
│   ├── ask_agent.py
│   ├── code_generator.py
│   ├── command_parser.py
│   ├── intent_router.py
│   ├── loader.py
│   ├── plan_executor.py
│   ├── plan_validator.py
│   ├── relationship_analysis.py
│   ├── summarizer.py
│   ├── type_inference.py
│   ├── utils.py
│   └── visualizations.py
├── tests/
│   ├── fixtures/
│   └── test_*.py
├── evaluation/
│   └── run_eval.py
├── sample_data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Future improvements

- Add configurable date parsing and time-series profiling.
- Support export to PDF and additional chart types.
- Add user-selectable sampling for very large datasets.
- Add more provider integrations and configurable privacy controls.
