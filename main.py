"""Command-line entry point for the sales regression baseline."""

from __future__ import annotations

import argparse
import json
import sys

from src.baseline import DataValidationError, inspect_dataset, predict_file, train_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect sales data, train a baseline model, or export predictions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Summarise a CSV or ZIP-contained CSV"
    )
    inspect_parser.add_argument("--input", required=True, help="Path to a CSV or ZIP archive")
    inspect_parser.add_argument(
        "--csv-name", help="CSV member name when the ZIP contains multiple CSV files"
    )

    train_parser = subparsers.add_parser(
        "train", help="Train and evaluate a mixed-type regression baseline"
    )
    train_parser.add_argument("--input", required=True, help="Path to a CSV or ZIP archive")
    train_parser.add_argument("--target", required=True, help="Numeric target column")
    train_parser.add_argument(
        "--date-column", help="Optional date column for chronological validation"
    )
    train_parser.add_argument(
        "--csv-name", help="CSV member name when the ZIP contains multiple CSV files"
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--max-iter", type=int, default=120)
    train_parser.add_argument("--model-out", default="artifacts/model.joblib")
    train_parser.add_argument("--metadata-out", default="artifacts/metrics.json")

    predict_parser = subparsers.add_parser(
        "predict", help="Generate predictions using a saved model"
    )
    predict_parser.add_argument("--input", required=True, help="Path to a CSV or ZIP archive")
    predict_parser.add_argument("--model", required=True, help="Path to a saved .joblib model")
    predict_parser.add_argument("--output", default="artifacts/predictions.csv")
    predict_parser.add_argument(
        "--csv-name", help="CSV member name when the ZIP contains multiple CSV files"
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "inspect":
            result = inspect_dataset(args.input, args.csv_name)
        elif args.command == "train":
            result = train_baseline(
                args.input,
                args.target,
                args.model_out,
                args.metadata_out,
                csv_name=args.csv_name,
                date_column=args.date_column,
                test_size=args.test_size,
                random_state=args.random_state,
                max_iter=args.max_iter,
            ).as_dict()
        else:
            result = predict_file(
                args.input,
                args.model,
                args.output,
                csv_name=args.csv_name,
            )
    except (DataValidationError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
