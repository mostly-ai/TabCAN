#!/usr/bin/env python
"""
Benchmark: TabCAN on TabArena datasets.

Runs TabCAN as a classifier on the TabArena benchmark suite using OpenML.

Usage:
    # Quick test (5 tasks, 1 fold)
    python scripts/benchmark_tabarena.py --max-tasks 5 --max-epochs 20

    # Full benchmark (all tasks, fold 0)
    python scripts/benchmark_tabarena.py --max-epochs 100

    # Single task
    python scripts/benchmark_tabarena.py --task-id 363614

    # Multiple folds for proper evaluation
    python scripts/benchmark_tabarena.py --folds 0,1,2,3,4,5,6,7

Requirements:
    pip install openml scikit-learn

Output:
    - tabarena-results.csv: Results for all tasks (TabArena submission format)
    - tabarena-summary.txt: Summary statistics

Submission to TabArena:
    1. Run full benchmark: python scripts/benchmark_tabarena.py --folds 0,1,2,3,4,5,6,7
    2. Results are in benchmark_results/tabarena-results.csv
    3. Open issue at https://github.com/autogluon/tabarena with results
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tabcan import TabCAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
_LOG = logging.getLogger(__name__)


def load_tabarena_task_ids() -> list[int]:
    """Load all task IDs from TabArena benchmark suite."""
    import openml

    _LOG.info("Loading TabArena benchmark suite...")
    benchmark_suite = openml.study.get_suite("tabarena-v0.1")
    task_ids = list(benchmark_suite.tasks)
    _LOG.info(f"Found {len(task_ids)} tasks in TabArena")
    return task_ids


def load_openml_task(task_id: int, fold: int = 0, repeat: int = 0) -> dict:
    """
    Load a single OpenML task with train/test split.

    Returns dict with X_train, y_train, X_test, y_test, task_name, problem_type
    """
    import openml

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )

    # Get train/test indices for this fold
    train_indices, test_indices = task.get_train_test_split_indices(
        fold=fold, repeat=repeat
    )

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    # Determine problem type
    task_type = task.task_type_id
    n_unique = y.nunique()
    if task_type == 2:  # Supervised Regression
        problem_type = "regression"
    elif task_type == 1:  # Supervised Classification
        problem_type = "binary" if n_unique == 2 else "multiclass"
    else:
        # Heuristic: if target has too many unique values, treat as regression
        if n_unique > 50:
            problem_type = "regression"
        else:
            problem_type = "binary" if n_unique == 2 else "multiclass"

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "task_name": dataset.name,
        "task_id": task_id,
        "problem_type": problem_type,
        "n_classes": y.nunique() if problem_type != "regression" else None,
        "categorical_indicator": categorical_indicator,
    }


def prepare_dataframe_for_tabcan(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str = "target",
    max_cardinality: int = 100,
    max_columns: int = 60,
) -> pd.DataFrame:
    """
    Prepare DataFrame for TabCAN training.

    - Converts all columns to string (categorical)
    - Bins high-cardinality columns
    - Limits number of columns
    - Puts target column LAST (required for prediction)
    """
    df = X.copy()

    # Convert all columns to object dtype first (avoid Categorical issues)
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].astype(str)

    # Limit number of columns (keep first N)
    if len(df.columns) > max_columns:
        df = df.iloc[:, :max_columns]

    # Convert numeric columns to binned categories
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            n_unique = df[col].nunique()
            if n_unique > max_cardinality:
                # Bin into quantiles
                try:
                    df[col] = pd.qcut(df[col], q=10, duplicates="drop").astype(str)
                except ValueError:
                    # If qcut fails, use regular cut
                    df[col] = pd.cut(df[col], bins=10).astype(str)
            else:
                df[col] = df[col].astype(str)
        else:
            # Convert to string first
            df[col] = df[col].astype(str)
            # Reduce cardinality for high-cardinality columns
            n_unique = df[col].nunique()
            if n_unique > max_cardinality:
                top_values = df[col].value_counts().nlargest(max_cardinality - 1).index
                df[col] = df[col].where(df[col].isin(top_values), "Other")

    # Handle missing values
    df = df.fillna("missing")

    # Add target as LAST column
    df[target_name] = y.astype(str)

    return df


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: list
) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    metrics = {"accuracy": accuracy_score(y_true, y_pred)}

    # Log loss
    try:
        # Need to align class probabilities with true labels
        y_true_encoded = np.array([classes.index(str(v)) for v in y_true])
        metrics["log_loss"] = log_loss(y_true_encoded, y_proba, labels=range(len(classes)))
    except Exception as e:
        _LOG.warning(f"Could not compute log_loss: {e}")
        metrics["log_loss"] = np.nan

    # ROC AUC (binary only)
    if len(classes) == 2:
        try:
            y_true_binary = (np.array([str(v) for v in y_true]) == classes[1]).astype(int)
            metrics["roc_auc"] = roc_auc_score(y_true_binary, y_proba[:, 1])
        except Exception as e:
            _LOG.warning(f"Could not compute roc_auc: {e}")
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def run_tabcan_on_task(
    task_data: dict,
    max_epochs: int = 100,
    patience: int = 20,
    embedding_mode: str = "text",
    verbose: int = 0,
) -> dict:
    """
    Run TabCAN on a single task.

    Returns dict with metrics and timing.
    """
    task_name = task_data["task_name"]
    task_id = task_data["task_id"]
    problem_type = task_data["problem_type"]

    _LOG.info(f"Running TabCAN on {task_name} (task_id={task_id}, type={problem_type})")

    if problem_type == "regression":
        _LOG.warning(f"Skipping regression task: {task_name}")
        return {"task_id": task_id, "task_name": task_name, "status": "skipped_regression"}

    # Prepare data
    target_col = "target"
    train_df = prepare_dataframe_for_tabcan(
        task_data["X_train"], task_data["y_train"], target_col
    )
    test_df = prepare_dataframe_for_tabcan(
        task_data["X_test"], task_data["y_test"], target_col
    )

    n_train = len(train_df)
    n_test = len(test_df)
    n_cols = len(train_df.columns)
    n_classes = task_data["n_classes"]

    _LOG.info(f"  Train: {n_train} rows, Test: {n_test} rows, Cols: {n_cols}, Classes: {n_classes}")

    # Train TabCAN
    try:
        start_time = time.time()

        model = TabCAN(
            max_epochs=max_epochs,
            patience=patience,
            embedding_mode=embedding_mode,
            verbose=verbose,
        )
        model.fit({"task": train_df})

        train_time = time.time() - start_time

        # Predict
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values

        start_time = time.time()
        y_proba = model.predict_proba(X_test, target_column=target_col, table_id="task")
        y_pred = model.predict(X_test, target_column=target_col, table_id="task")
        predict_time = time.time() - start_time

        classes = model.get_classes(target_column=target_col, table_id="task")

        # Evaluate
        metrics = evaluate_classification(y_test, y_pred, y_proba, classes)

        result = {
            "task_id": task_id,
            "task_name": task_name,
            "problem_type": problem_type,
            "n_train": n_train,
            "n_test": n_test,
            "n_cols": n_cols,
            "n_classes": n_classes,
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "roc_auc": metrics["roc_auc"],
            "train_time": train_time,
            "predict_time": predict_time,
            "status": "success",
        }

        _LOG.info(
            f"  Results: accuracy={metrics['accuracy']:.4f}, "
            f"log_loss={metrics['log_loss']:.4f}, "
            f"train_time={train_time:.1f}s"
        )

        return result

    except Exception as e:
        _LOG.error(f"  Error on {task_name}: {e}")
        return {
            "task_id": task_id,
            "task_name": task_name,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TabCAN on TabArena")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to run (None = all)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Run a specific task ID only",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="text",
        choices=["text", "vocab"],
        help="Embedding mode",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0",
        help="Comma-separated list of folds to run (e.g., '0,1,2,3,4,5,6,7' for full CV)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )
    args = parser.parse_args()

    # Parse folds
    folds = [int(f.strip()) for f in args.folds.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get task IDs
    if args.task_id:
        task_ids = [args.task_id]
    else:
        task_ids = load_tabarena_task_ids()
        if args.max_tasks:
            task_ids = task_ids[: args.max_tasks]

    _LOG.info(f"Running TabCAN on {len(task_ids)} tasks, {len(folds)} fold(s)")

    # Run on each task and fold
    results = []
    total_runs = len(task_ids) * len(folds)
    run_idx = 0

    for task_id in task_ids:
        for fold in folds:
            run_idx += 1
            _LOG.info(f"\n=== Run {run_idx}/{total_runs}: Task {task_id}, Fold {fold} ===")

            try:
                task_data = load_openml_task(task_id, fold=fold)
                result = run_tabcan_on_task(
                    task_data,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    embedding_mode=args.embedding_mode,
                    verbose=args.verbose,
                )
                result["fold"] = fold
                results.append(result)

                # Save intermediate results
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_dir / "tabarena-results.csv", index=False)

            except Exception as e:
                _LOG.error(f"Failed on task {task_id}, fold {fold}: {e}")
                results.append({
                    "task_id": task_id,
                    "fold": fold,
                    "status": "load_error",
                    "error": str(e),
                })

    # Final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "tabarena-results.csv", index=False)

    # Summary
    _LOG.info("\n" + "=" * 60)
    _LOG.info("TABARENA BENCHMARK RESULTS - TabCAN")
    _LOG.info("=" * 60)

    successful = results_df[results_df["status"] == "success"]
    skipped = results_df[results_df["status"].str.contains("skipped", na=False)]
    failed = results_df[~results_df["status"].isin(["success"]) & ~results_df["status"].str.contains("skipped", na=False)]

    _LOG.info(f"\nTasks: {len(task_ids)}, Folds: {len(folds)}")
    _LOG.info(f"Successful runs: {len(successful)}")
    _LOG.info(f"Skipped (regression): {len(skipped)}")
    _LOG.info(f"Failed: {len(failed)}")

    if len(successful) > 0:
        _LOG.info(f"\n--- Aggregated Metrics (across all folds) ---")
        _LOG.info(f"Mean accuracy: {successful['accuracy'].mean():.4f} (+/- {successful['accuracy'].std():.4f})")
        _LOG.info(f"Mean log_loss: {successful['log_loss'].mean():.4f} (+/- {successful['log_loss'].std():.4f})")
        if successful["roc_auc"].notna().any():
            roc_vals = successful["roc_auc"].dropna()
            _LOG.info(f"Mean ROC AUC (binary): {roc_vals.mean():.4f} (+/- {roc_vals.std():.4f})")
        _LOG.info(f"Mean train time: {successful['train_time'].mean():.1f}s")
        _LOG.info(f"Mean predict time: {successful['predict_time'].mean():.1f}s")

        # Per-task summary
        _LOG.info(f"\n--- Per-Task Summary ---")
        task_summary = successful.groupby("task_name").agg({
            "accuracy": ["mean", "std"],
            "log_loss": ["mean", "std"],
            "roc_auc": ["mean", "std"],
        }).round(4)
        print(task_summary.to_string())

    # Write summary file
    summary_path = output_dir / "tabarena-summary.txt"
    with open(summary_path, "w") as f:
        f.write("TabArena Benchmark Results - TabCAN\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: TabCAN\n")
        f.write(f"Embedding mode: {args.embedding_mode}\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Patience: {args.patience}\n")
        f.write(f"Tasks: {len(task_ids)}\n")
        f.write(f"Folds: {folds}\n\n")

        if len(successful) > 0:
            f.write("Aggregated Results:\n")
            f.write(f"  Accuracy: {successful['accuracy'].mean():.4f} (+/- {successful['accuracy'].std():.4f})\n")
            f.write(f"  Log Loss: {successful['log_loss'].mean():.4f} (+/- {successful['log_loss'].std():.4f})\n")
            if successful["roc_auc"].notna().any():
                roc_vals = successful["roc_auc"].dropna()
                f.write(f"  ROC AUC: {roc_vals.mean():.4f} (+/- {roc_vals.std():.4f})\n")
            f.write(f"  Train time: {successful['train_time'].mean():.1f}s\n")

    _LOG.info(f"\nResults saved to:")
    _LOG.info(f"  {output_dir / 'tabarena-results.csv'}")
    _LOG.info(f"  {output_dir / 'tabarena-summary.txt'}")


if __name__ == "__main__":
    main()
