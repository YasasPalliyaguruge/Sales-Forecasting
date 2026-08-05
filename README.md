# Sales Forecasting Baseline

![Sales Forecasting project cover](assets/recruiter/cover.png)

A small, reproducible command-line workflow for inspecting tabular sales data, training a mixed-type regression baseline and exporting predictions. The project accepts either a CSV file or a ZIP archive containing CSV data and does not assume a fixed retail schema.

## What is implemented

- Data-quality inspection: shape, column types, missing values and duplicates.
- Explicit target-column validation.
- Automatic numeric and categorical preprocessing.
- Missing-value imputation and unseen-category handling.
- Optional date-derived features and chronological holdout evaluation.
- Random holdout evaluation when no date column is supplied.
- MAE, RMSE and R² metrics.
- Saved `joblib` model plus JSON metadata.
- Batch prediction export to CSV.
- Automated tests for CSV/ZIP loading, validation, training and prediction.

This is a supervised regression baseline. It is not yet a full time-series forecasting platform, experiment-tracking service or Flask application.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Inspect a dataset

```bash
python main.py inspect --input path/to/training_data.csv
```

For a ZIP containing one CSV:

```bash
python main.py inspect --input path/to/training_data.zip
```

When an archive contains multiple CSV files, select one explicitly:

```bash
python main.py inspect --input data.zip --csv-name training_data.csv
```

## Train a baseline

Choose the numeric sales target explicitly:

```bash
python main.py train \
  --input path/to/training_data.csv \
  --target sales
```

When the dataset has a date column, use a chronological validation split:

```bash
python main.py train \
  --input path/to/training_data.csv \
  --target sales \
  --date-column date \
  --model-out artifacts/model.joblib \
  --metadata-out artifacts/metrics.json
```

The date column is sorted before the split and converted into year, month, day and weekday features. Without `--date-column`, the project uses a repeatable random holdout.

## Export predictions

```bash
python main.py predict \
  --input path/to/test_data.csv \
  --model artifacts/model.joblib \
  --output artifacts/predictions.csv
```

Prediction data must contain the same raw feature columns used for training. Extra columns are preserved in the output.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the test suite and a CLI smoke test for pull requests and pushes to `main`.

## Data repository

The separate `Data-Sales_Forecasting` repository currently stores compressed training, test and outlet files. Their licence and final distribution path should be confirmed before merging or republishing them here. This project deliberately leaves those archives unchanged.

## Current limitations

- The real dataset target and date columns must be selected by the person running the project.
- The baseline uses one holdout split rather than rolling-origin cross-validation.
- It does not yet provide model comparison, hyperparameter search, MLflow tracking or a deployed prediction API.
- The model predicts a numeric target from tabular features; a production forecasting system would require business-specific temporal validation, leakage review and monitoring.
