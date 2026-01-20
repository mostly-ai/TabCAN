#!/usr/bin/env python
"""
Download and cache T4 dataset locally.

The T4 dataset contains 3.1M tables from TabLib.
https://huggingface.co/datasets/mlfoundations/t4-full

This script downloads tables from HuggingFace and caches them as
preprocessed parquet files for efficient training.

Usage:
    # Download first 1000 tables
    python scripts/download_t4.py --num_tables 1000

    # Download all tables (requires HF_TOKEN for gated dataset)
    HF_TOKEN=your_token python scripts/download_t4.py

    # Specify custom cache directory
    python scripts/download_t4.py --cache_dir /data/t4_cache --num_tables 1000
"""

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOG = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "~/.cache/tabcan/t4"


def download_t4(
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_tables: int | None = None,
    categorical_only: bool = True,
    min_columns: int = 2,
    max_columns: int = 64,
    max_rows: int = 10000,
) -> dict:
    """
    Download T4 dataset to local cache.

    Args:
        cache_dir: Directory to cache tables
        num_tables: Maximum number of tables to download (None = all)
        categorical_only: Only include categorical columns
        min_columns: Minimum columns required
        max_columns: Maximum columns to include
        max_rows: Maximum rows per table

    Returns:
        Index dictionary with metadata for each table
    """
    cache_path = Path(cache_dir).expanduser()
    tables_dir = cache_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists
    index_path = cache_path / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            existing_index = json.load(f)
        existing_count = len(existing_index)
        _LOG.info(f"Found existing cache with {existing_count} tables")
        if num_tables is not None and existing_count >= num_tables:
            _LOG.info("Cache already has enough tables, skipping download")
            return existing_index
        _LOG.info(f"Will continue downloading from table {existing_count}")
        start_table_idx = existing_count
        index = existing_index
    else:
        start_table_idx = 0
        index = {}

    fs = HfFileSystem()

    # List all chunk zip files
    dataset_path = "datasets/mlfoundations/t4-full"
    try:
        files = fs.ls(dataset_path, detail=False)
    except Exception as e:
        _LOG.error(f"Failed to list T4 dataset. Make sure HF_TOKEN is set. Error: {e}")
        raise

    # Filter to zip files containing parquet
    zip_files = sorted([f for f in files if f.endswith(".zip")])
    _LOG.info(f"Found {len(zip_files)} zip files in T4 dataset")

    table_idx = start_table_idx
    tables_downloaded = 0

    for zip_path in zip_files:
        if num_tables is not None and table_idx >= num_tables:
            break

        _LOG.info(f"Processing {zip_path}")

        try:
            # Download and open zip file
            with fs.open(zip_path, "rb") as f:
                zip_data = f.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                # Find parquet files in zip
                parquet_files = sorted(
                    [n for n in zf.namelist() if n.endswith(".parquet")]
                )

                for pq_name in parquet_files:
                    if num_tables is not None and table_idx >= num_tables:
                        break

                    try:
                        # Read parquet file from zip
                        with zf.open(pq_name) as pq_file:
                            table = pq.read_table(io.BytesIO(pq_file.read()))
                            df = table.to_pandas()

                        # Filter columns
                        if categorical_only:
                            cat_cols = df.select_dtypes(
                                include=["object", "string", "category"]
                            ).columns.tolist()
                            df = df[cat_cols]

                        # Skip if too few columns
                        if len(df.columns) < min_columns:
                            continue

                        # Truncate columns
                        if len(df.columns) > max_columns:
                            df = df.iloc[:, :max_columns]

                        # Truncate rows
                        if len(df) > max_rows:
                            df = df.head(max_rows)

                        # Convert all to string
                        df = df.astype(str)

                        # Save preprocessed table
                        table_path = tables_dir / f"{table_idx:06d}.parquet"
                        df.to_parquet(table_path, index=False)

                        # Record in index
                        index[str(table_idx)] = {
                            "path": f"tables/{table_idx:06d}.parquet",
                            "num_rows": len(df),
                            "num_columns": len(df.columns),
                            "columns": df.columns.tolist(),
                        }

                        table_idx += 1
                        tables_downloaded += 1

                        if tables_downloaded % 100 == 0:
                            _LOG.info(
                                f"Downloaded {tables_downloaded} tables "
                                f"(total: {table_idx})"
                            )
                            # Save index periodically
                            with open(index_path, "w") as f:
                                json.dump(index, f, indent=2)

                    except Exception as e:
                        _LOG.warning(f"Failed to process {pq_name}: {e}")
                        continue

        except Exception as e:
            _LOG.warning(f"Failed to process {zip_path}: {e}")
            continue

    # Save final index
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    _LOG.info(f"Download complete. Total tables: {len(index)}")
    _LOG.info(f"Cache directory: {cache_path}")

    # Print stats
    total_rows = sum(meta["num_rows"] for meta in index.values())
    _LOG.info(f"Total rows across all tables: {total_rows:,}")

    return index


def main():
    parser = argparse.ArgumentParser(description="Download and cache T4 dataset")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--num_tables",
        type=int,
        default=None,
        help="Number of tables to download (default: all)",
    )
    parser.add_argument(
        "--max_columns",
        type=int,
        default=64,
        help="Maximum columns per table (default: 64)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=10000,
        help="Maximum rows per table (default: 10000)",
    )
    args = parser.parse_args()

    download_t4(
        cache_dir=args.cache_dir,
        num_tables=args.num_tables,
        max_columns=args.max_columns,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
