"""Reusable baseline training and prediction utilities for tabular sales data."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class DataValidationError(ValueError):
    """Raised when an input dataset cannot be used safely."""


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, float]
    rows: int
    training_rows: int
    validation_rows: int
    feature_count: int
    model_path: str
    metadata_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "rows": self.rows,
            "training_rows": self.training_rows,
            "validation_rows": self.validation_rows,
            "feature_count": self.feature_count,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
        }


def load_table(input_path: str | Path, csv_name: str | None = None) -> pd.DataFrame:
    """Load a CSV directly or from a ZIP archive containing CSV files."""
    path = Path(input_path)
    if not path.exists():
        raise DataValidationError(f"Input file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)

    if path.suffix.lower() != ".zip":
        raise DataValidationError("Input must be a .csv file or a .zip archive containing CSV data")

    try:
        with zipfile.ZipFile(path) as archive:
            csv_members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            ]
            if not csv_members:
                raise DataValidationError(f"ZIP archive contains no CSV files: {path}")

            if csv_name:
                matches = [name for name in csv_members if name == csv_name or Path(name).name == csv_name]
                if len(matches) != 1:
                    raise DataValidationError(
                        f"Could not uniquely resolve CSV '{csv_name}'. Available files: {', '.join(csv_members)}"
                    )
                selected = matches[0]
            elif len(csv_members) == 1:
                selected = csv_members[0]
            else:
                raise DataValidationError(
                    "ZIP archive contains multiple CSV files. Use --csv-name to select one: "
                    + ", ".join(csv_members)
                )

            with archive.open(selected) as handle:
                return pd.read_csv(handle, low_memory=False)
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"Invalid ZIP archive: {path}") from exc


def inspect_dataset(input_path: str | Path, csv_name: str | None = None) -> dict[str, Any]:
    """Return a JSON-serialisable data-quality summary."""
    frame = load_table(input_path, csv_name)
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(column) for column in frame.columns],
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in frame.columns if column not in numeric_columns],
        "missing_values": {
            str(column): int(count) for column, count in frame.isna().sum().items() if count
        },
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def _prepare_features(frame: pd.DataFrame, date_column: str | None) -> pd.DataFrame:
    prepared = frame.copy()
    if not date_column:
        return prepared
    if date_column not in prepared.columns:
        raise DataValidationError(f"Date column not found: {date_column}")

    parsed = pd.to_datetime(prepared[date_column], errors="coerce", utc=True)
    invalid = int(parsed.isna().sum())
    if invalid:
        raise DataValidationError(f"Date column '{date_column}' contains {invalid} unparseable value(s)")

    prepared[f"{date_column}__year"] = parsed.dt.year
    prepared[f"{date_column}__month"] = parsed.dt.month
    prepared[f"{date_column}__day"] = parsed.dt.day
    prepared[f"{date_column}__dayofweek"] = parsed.dt.dayofweek
    return prepared.drop(columns=[date_column])


def _build_pipeline(features: pd.DataFrame, random_state: int, max_iter: int) -> Pipeline:
    numeric_columns = list(features.select_dtypes(include=[np.number]).columns)
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    if not numeric_columns and not categorical_columns:
        raise DataValidationError("No feature columns remain after removing the target")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=random_state,
                    learning_rate=0.05,
                    max_iter=max_iter,
                    l2_regularization=1.0,
                ),
            ),
        ]
    )


def train_baseline(
    input_path: str | Path,
    target_column: str,
    model_path: str | Path,
    metadata_path: str | Path,
    *,
    csv_name: str | None = None,
    date_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 120,
) -> TrainingResult:
    """Train a mixed-type regression baseline and persist the fitted artifact."""
    if not 0.05 <= test_size <= 0.5:
        raise DataValidationError("test_size must be between 0.05 and 0.5")

    frame = load_table(input_path, csv_name)
    if target_column not in frame.columns:
        raise DataValidationError(f"Target column not found: {target_column}")
    if date_column == target_column:
        raise DataValidationError("The date column and target column must be different")
    if len(frame) < 20:
        raise DataValidationError("At least 20 rows are required to train and evaluate the baseline")

    target = pd.to_numeric(frame[target_column], errors="coerce")
    invalid_targets = int(target.isna().sum())
    if invalid_targets:
        raise DataValidationError(
            f"Target column '{target_column}' contains {invalid_targets} missing or non-numeric value(s)"
        )

    raw_features = frame.drop(columns=[target_column])
    if date_column:
        parsed_dates = pd.to_datetime(raw_features[date_column], errors="coerce", utc=True)
        invalid_dates = int(parsed_dates.isna().sum())
        if invalid_dates:
            raise DataValidationError(
                f"Date column '{date_column}' contains {invalid_dates} unparseable value(s)"
            )
        order = np.argsort(parsed_dates.to_numpy())
        raw_features = raw_features.iloc[order].reset_index(drop=True)
        target = target.iloc[order].reset_index(drop=True)

    features = _prepare_features(raw_features, date_column)
    validation_rows = max(1, int(math.ceil(len(features) * test_size)))

    if date_column:
        split_index = len(features) - validation_rows
        x_train, x_validation = features.iloc[:split_index], features.iloc[split_index:]
        y_train, y_validation = target.iloc[:split_index], target.iloc[split_index:]
        split_strategy = "chronological"
    else:
        rng = np.random.default_rng(random_state)
        indices = np.arange(len(features))
        rng.shuffle(indices)
        validation_indices = indices[:validation_rows]
        training_indices = indices[validation_rows:]
        x_train, x_validation = features.iloc[training_indices], features.iloc[validation_indices]
        y_train, y_validation = target.iloc[training_indices], target.iloc[validation_indices]
        split_strategy = "random"

    pipeline = _build_pipeline(x_train, random_state=random_state, max_iter=max_iter)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_validation)

    metrics = {
        "mae": float(mean_absolute_error(y_validation, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_validation, predictions))),
        "r2": float(r2_score(y_validation, predictions)),
    }

    artifact = {
        "pipeline": pipeline,
        "target_column": target_column,
        "date_column": date_column,
        "raw_feature_columns": [str(column) for column in raw_features.columns],
        "prepared_feature_columns": [str(column) for column in features.columns],
        "metrics": metrics,
        "split_strategy": split_strategy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    model_output = Path(model_path)
    metadata_output = Path(metadata_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output)

    metadata = {key: value for key, value in artifact.items() if key != "pipeline"}
    metadata.update(
        {
            "rows": int(len(frame)),
            "training_rows": int(len(x_train)),
            "validation_rows": int(len(x_validation)),
            "feature_count": int(len(features.columns)),
        }
    )
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingResult(
        metrics=metrics,
        rows=int(len(frame)),
        training_rows=int(len(x_train)),
        validation_rows=int(len(x_validation)),
        feature_count=int(len(features.columns)),
        model_path=str(model_output),
        metadata_path=str(metadata_output),
    )


def predict_file(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    csv_name: str | None = None,
) -> dict[str, Any]:
    """Load a fitted artifact, generate predictions and write them to CSV."""
    artifact = joblib.load(model_path)
    required_keys = {"pipeline", "date_column", "raw_feature_columns"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise DataValidationError("Model artifact is not a supported Sales-Forecasting baseline")

    frame = load_table(input_path, csv_name)
    required_columns = list(artifact["raw_feature_columns"])
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(
            "Prediction data is missing required columns: " + ", ".join(missing)
        )

    raw_features = frame[required_columns]
    prepared = _prepare_features(raw_features, artifact.get("date_column"))
    predictions = artifact["pipeline"].predict(prepared)

    output = frame.copy()
    output["prediction"] = predictions
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return {"rows": int(len(output)), "output_path": str(destination)}
