import math
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline import (
    DataValidationError,
    inspect_dataset,
    load_table,
    predict_file,
    train_baseline,
)


class BaselinePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        rows = 60
        dates = pd.date_range("2025-01-01", periods=rows, freq="D")
        region = np.where(np.arange(rows) % 2 == 0, "north", "south")
        promotion = np.arange(rows) % 3 == 0
        units = 20 + np.arange(rows) * 0.5
        noise = np.sin(np.arange(rows))
        sales = units * 12 + promotion.astype(int) * 35 + (region == "north") * 10 + noise
        self.frame = pd.DataFrame(
            {
                "date": dates.astype(str),
                "region": region,
                "promotion": promotion,
                "units": units,
                "sales": sales,
            }
        )
        self.csv_path = self.root / "sales.csv"
        self.frame.to_csv(self.csv_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_inspects_csv(self) -> None:
        summary = inspect_dataset(self.csv_path)
        self.assertEqual(summary["rows"], 60)
        self.assertIn("sales", summary["numeric_columns"])

    def test_loads_single_csv_from_zip(self) -> None:
        archive_path = self.root / "sales.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(self.csv_path, arcname="nested/sales.csv")
        loaded = load_table(archive_path)
        self.assertEqual(list(loaded.columns), list(self.frame.columns))

    def test_requires_csv_selection_for_ambiguous_zip(self) -> None:
        archive_path = self.root / "multiple.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("one.csv", "a\n1\n")
            archive.writestr("two.csv", "a\n2\n")
        with self.assertRaises(DataValidationError):
            load_table(archive_path)

    def test_trains_and_exports_predictions(self) -> None:
        model_path = self.root / "model.joblib"
        metadata_path = self.root / "metrics.json"
        result = train_baseline(
            self.csv_path,
            "sales",
            model_path,
            metadata_path,
            date_column="date",
            max_iter=30,
        )
        self.assertTrue(model_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(result.rows, 60)
        self.assertIn("rmse", result.metrics)

        prediction_input = self.frame.drop(columns=["sales"]).tail(5)
        prediction_path = self.root / "prediction_input.csv"
        prediction_input.to_csv(prediction_path, index=False)
        output_path = self.root / "predictions.csv"
        prediction_result = predict_file(prediction_path, model_path, output_path)
        self.assertEqual(prediction_result["rows"], 5)
        output = pd.read_csv(output_path)
        self.assertIn("prediction", output.columns)

    def test_rejects_non_numeric_target(self) -> None:
        invalid = self.frame.copy()
        invalid["sales"] = "unknown"
        invalid_path = self.root / "invalid.csv"
        invalid.to_csv(invalid_path, index=False)
        with self.assertRaises(DataValidationError):
            train_baseline(
                invalid_path,
                "sales",
                self.root / "m.joblib",
                self.root / "m.json",
            )

    def test_rejects_missing_date_column_cleanly(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "Date column not found"):
            train_baseline(
                self.csv_path,
                "sales",
                self.root / "missing-date.joblib",
                self.root / "missing-date.json",
                date_column="missing_date",
            )

    def test_minimum_holdout_has_two_rows_and_finite_metrics(self) -> None:
        small_path = self.root / "small.csv"
        self.frame.head(20).to_csv(small_path, index=False)
        result = train_baseline(
            small_path,
            "sales",
            self.root / "small.joblib",
            self.root / "small.json",
            date_column="date",
            test_size=0.05,
            max_iter=20,
        )
        self.assertEqual(result.validation_rows, 2)
        self.assertTrue(all(math.isfinite(value) for value in result.metrics.values()))


if __name__ == "__main__":
    unittest.main()
