# AI-Powered-CSV-Data-Analyst

A local Streamlit app that profiles uploaded CSV files, automatically selects
useful visualizations, and explains the results in plain English. It works
without an AI provider key; optional OpenAI or Anthropic keys power the dataset
summary explanation.

## Features

- Upload `.csv` files up to 50 MB with delimiter and common encoding detection.
- Review dataset dimensions, memory use, duplicates, types, and a 10-row preview.
- Explore missingness, numeric statistics, categorical counts, skewness, and
  IQR-based outliers.
- Automatically scan the dataset and select useful charts for distributions,
  numeric relationships, categories, date trends, and missing values.
- Read short data-quality and statistical findings alongside the charts.
- Download charts and the complete profile as TXT or Markdown.
- Keep analysis local except for compact aggregate summaries sent to an optional
  AI provider; raw rows and category labels are never sent.

## Data Story

The Data Story is an automatic exploratory analysis, not a chart-type picker.
It infers column roles and selects up to six charts that fit the dataset:

- Date plus numeric measure: trend line and rolling mean.
- Numeric measures: distributions, correlation heatmap, and a scatterplot for
  a sufficiently strong numeric association.
- Numeric measure plus group: grouped boxplot.
- Low-cardinality category: most-common-values bar chart.
- Incomplete columns: missingness chart.

The report also highlights missing values, duplicates, skew, and strong
correlations. Identifier-like, constant, empty, and high-cardinality fields are
not treated as useful measures or groups. Large row-heavy charts use a
reproducible sample, and the app discloses the sample size.

## Setup

Python 3.10 or newer is required.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload either CSV in `sample_data/` to explore the app.

## Optional AI explanation

Copy `.env.example` to `.env`, choose `AI_PROVIDER=openai` or `anthropic`, and
set the matching `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The model can be
selected with `OPENAI_MODEL` or `ANTHROPIC_MODEL`. Keep `.env` private; it is
ignored by Git. Without a key, the app provides a rule-based explanation.

## Tests

```powershell
python -m pytest
```

The tests cover upload handling, analysis, Data Story chart selection, AI
explanation fallback, report downloads, and Streamlit upload/rendering.

## Project structure

```text
AI-Powered-CSV-Data-Analyst/
├── app.py
├── src/
│   ├── ai_explainer.py
│   ├── analysis.py
│   ├── auto_visualizer.py
│   ├── loader.py
│   ├── type_inference.py
│   ├── utils.py
│   └── visualizations.py
├── tests/
│   ├── fixtures/
│   └── test_*.py
├── sample_data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitations

Automatic chart selection uses deterministic rules and aggregate statistics;
it does not infer domain-specific meaning beyond column names and data types.
AI explanations are optional and depend on the configured provider. When used,
only compact aggregates and schema information are sent. The app is designed
for local use with the user's own data.
