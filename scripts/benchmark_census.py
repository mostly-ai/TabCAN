#!/usr/bin/env python
"""
Benchmark: TabCAN performance on Census dataset.

Usage:
    python scripts/benchmark_census.py

Produces:
    - census-results.csv: Results table
    - census-results.png: Plot of results
    - syn-tabcan-{n}rows.csv: Synthetic data for each size
    - model-tabcan-{n}rows.html: QA reports
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


def load_census_data():
    """Load and preprocess census data."""
    _LOG.info("Loading census data...")
    df = pd.read_csv(
        "https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz"
    ).sample(frac=1, random_state=42)

    # Select columns
    cols = [
        c
        for c in df.columns
        if c
        in [
            "workclass",
            "education",
            "marital_status",
            "relationship",
            "race",
            "sex",
            "income",
            "native_country",
            "age",
            "hours_per_week",
        ]
    ]
    df = df[cols]

    # Reduce cardinality
    for col, n in [("education", 8), ("native_country", 8)]:
        top = df[col].value_counts().nlargest(n).index
        df[col] = df[col].where(df[col].isin(top), "Other")

    # Bucket age
    bins = [0, 25, 35, 45, 55, 65, 75, 99]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["age"] = pd.cut(df["age"], bins=bins, labels=labels).astype(str)

    # Bucket hours per week
    bins = [0, 25, 39, 41, 50, 99]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["hours_per_week"] = pd.cut(df["hours_per_week"], bins=bins, labels=labels).astype(
        str
    )

    _LOG.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def evaluate(model_name, n, trn_df, hol_df, syn_df, output_dir):
    """Evaluate synthetic data quality using mostlyai.qa."""
    from mostlyai import qa

    report_path, metrics = qa.report(
        syn_tgt_data=syn_df,
        trn_tgt_data=trn_df,
        hol_tgt_data=hol_df,
        report_path=str(output_dir / f"model-{model_name}-{n}rows.html"),
    )
    syn_df.to_csv(output_dir / f"syn-{model_name}-{n}rows.csv", index=False)

    return {
        "model": model_name,
        "rows": n,
        "accuracy": metrics.accuracy.overall,
        "univariate": metrics.accuracy.univariate,
        "bivariate": metrics.accuracy.bivariate,
        "trivariate": metrics.accuracy.trivariate,
        "dcr_share": metrics.distances.dcr_share,
    }


def run_tabcan(df, n, output_dir, model_size="medium", max_epochs=100, patience=10, random_state=42, checkpoint=None, embedding_mode="text"):
    """Run TabCAN benchmark."""
    _LOG.info(f"TabCAN ({model_size}, {embedding_mode}): n={n}")
    t0 = time.time()

    model = TabCAN(
        model_size=model_size,
        checkpoint=checkpoint,
        embedding_mode=embedding_mode,
        max_epochs=max_epochs,
        patience=patience,
        verbose=1,
        random_state=random_state,
    )
    model.fit({"census": df.iloc[:n]})

    syn_df = model.sample(n_samples=n, table_id="census")
    t1 = time.time()

    model_name = f"TabCAN-{model_size}-{embedding_mode}"
    row = evaluate(
        model_name,
        n,
        trn_df=df.iloc[:n],
        hol_df=df.iloc[-max(n, 1000):],
        syn_df=syn_df,
        output_dir=output_dir,
    )
    row["compute_time"] = t1 - t0
    return row


def plot_results(csv_path, output_path, dpi=300):
    """Plot benchmark results with 6 metrics in 2x3 grid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)

    # Color mappings
    colors = {
        "TabCAN-small": "#3FB9F2",
        "TabCAN-medium": "#278918",
        "TabCAN-large": "#F2A63F",
    }

    # Metrics configuration (metric_column, display_title)
    metrics = [
        ("accuracy", "Overall Accuracy"),
        ("univariate", "Univariate Accuracy"),
        ("dcr_share", "DCR Share"),
        ("bivariate", "Bivariate Accuracy"),
        ("trivariate", "Trivariate Accuracy"),
        ("compute_time", "Compute Time"),
    ]

    # Create plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for model in df["model"].unique():
            data = df[df["model"] == model].sort_values("rows")
            ax.plot(
                data["rows"],
                data[metric],
                marker="o",
                linewidth=2,
                markersize=8,
                color=colors.get(model, "gray"),
                label=model,
            )

        ax.set_xlabel("nrows", fontsize=11, fontweight="bold")
        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xscale("log")
        if metric == "compute_time":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax.set_xticks([100, 1000, 10000])
        ax.set_xticklabels(["100", "1000", "10000"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    _LOG.info(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TabCAN on Census dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=500,
        help="Max epochs for training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,1000,10000",
        help="Comma-separated list of dataset sizes to benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained checkpoint for fine-tuning",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="text",
        choices=["text", "vocab"],
        help="Embedding mode: 'text' (LLM) or 'vocab' (learnable)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse sizes
    sizes = [int(s) for s in args.sizes.split(",")]

    # Load data
    df = load_census_data()

    results = []

    for n in sizes:
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"Running TabCAN benchmark for n={n}")
        _LOG.info(f"{'='*60}")

        try:
            row = run_tabcan(
                df, n, output_dir, args.model_size, args.max_epochs, args.patience, args.seed, args.checkpoint, args.embedding_mode
            )
            results.append(row)
            _LOG.info(
                f"TabCAN-{args.model_size}-{args.embedding_mode}: accuracy={row['accuracy']:.3f}, time={row['compute_time']:.1f}s"
            )
        except Exception as e:
            _LOG.error(f"TabCAN failed: {e}")
            import traceback

            traceback.print_exc()

        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / f"census-results-{n}rows.csv", index=False)

    # Save final results
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "census-results.csv"
    results_df.to_csv(csv_path, index=False)
    _LOG.info(f"\nResults saved to {csv_path}")

    # Print results table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Plot results
    try:
        plot_results(csv_path, output_dir / "census-results.png")
    except Exception as e:
        _LOG.error(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
