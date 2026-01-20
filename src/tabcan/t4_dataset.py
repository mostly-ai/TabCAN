"""
PyTorch Dataset for cached T4 tables.

Provides efficient global random sampling across all tables with LRU caching.
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

_LOG = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tabcan" / "t4"


class T4Dataset(Dataset):
    """
    Dataset for pre-training on cached T4 tables.

    Features:
    - Global random sampling: samples uniformly across all rows of all tables
    - LRU cache: keeps frequently accessed tables in memory
    - Lazy loading: only loads tables when needed

    Args:
        cache_dir: Path to cached T4 data (with index.json)
        max_tables: Maximum number of tables to include (None = all)
        max_columns: Maximum columns per row
        cache_size: Number of tables to keep in memory (LRU cache)

    Example:
        >>> from tabcan.t4_dataset import T4Dataset
        >>>
        >>> dataset = T4Dataset()
        >>> print(f"Total rows: {len(dataset)}")
        >>>
        >>> # Random access
        >>> item = dataset[12345]
        >>> print(item["columns"], item["text_values"])
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_tables: int | None = None,
        max_columns: int = 64,
        cache_size: int = 1000,
    ):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser()
        self.max_columns = max_columns
        self._cache_size = cache_size

        # Load index
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                "Run scripts/download_t4.py first to download the dataset."
            )

        with open(index_path) as f:
            self.index = json.load(f)

        _LOG.info(f"Loaded index with {len(self.index)} tables")

        # Build global row index: [(table_id, row_idx), ...]
        self.row_index: list[tuple[int, int]] = []
        tables_included = 0
        total_rows = 0

        for table_id in sorted(self.index.keys(), key=int):
            if max_tables is not None and tables_included >= max_tables:
                break

            meta = self.index[table_id]
            num_rows = meta["num_rows"]

            for row_idx in range(num_rows):
                self.row_index.append((int(table_id), row_idx))

            tables_included += 1
            total_rows += num_rows

        _LOG.info(
            f"Built row index: {tables_included} tables, {total_rows:,} total rows"
        )

        # LRU cache for loaded tables
        self._table_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

        # Cache for table metadata (columns + class_values) - computed once per table
        self._metadata_cache: dict[int, dict] = {}

    def _load_table(self, table_id: int) -> pd.DataFrame:
        """Load a table from disk or cache."""
        if table_id in self._table_cache:
            # Move to end (most recently used)
            self._table_cache.move_to_end(table_id)
            return self._table_cache[table_id]

        # Load from disk
        meta = self.index[str(table_id)]
        path = self.cache_dir / meta["path"]
        df = pd.read_parquet(path)

        # Add to cache
        self._table_cache[table_id] = df

        # Evict oldest if cache full
        while len(self._table_cache) > self._cache_size:
            self._table_cache.popitem(last=False)

        return df

    def __len__(self) -> int:
        return len(self.row_index)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single row.

        Returns:
            Dict with:
            - text_values: List of string values for each column
            - columns: List of column names
            - table_id: Table ID
        """
        table_id, row_idx = self.row_index[idx]
        df = self._load_table(table_id)

        # Get columns (truncate if needed)
        columns = df.columns.tolist()[: self.max_columns]
        row = df.iloc[row_idx]

        # Convert to string values
        text_values = [str(row[col]) for col in columns]

        return {
            "text_values": text_values,
            "columns": columns,
            "table_id": table_id,
        }

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        num_tables = len(set(t for t, _ in self.row_index))
        total_rows = len(self.row_index)

        # Column stats from index
        all_num_columns = [
            self.index[str(t)]["num_columns"] for t in set(t for t, _ in self.row_index)
        ]

        return {
            "num_tables": num_tables,
            "total_rows": total_rows,
            "avg_columns": (
                sum(all_num_columns) / len(all_num_columns) if all_num_columns else 0
            ),
            "min_columns": min(all_num_columns) if all_num_columns else 0,
            "max_columns": max(all_num_columns) if all_num_columns else 0,
        }

    def get_table_ids(self) -> list[int]:
        """Get list of unique table IDs in the dataset."""
        return sorted(set(t for t, _ in self.row_index))

    def get_table_metadata(self, table_id: int) -> dict:
        """
        Get metadata for a table including columns and unique values.

        Metadata is computed once per table and cached.

        Args:
            table_id: Table ID to get metadata for

        Returns:
            Dict with:
            - columns: list of column names
            - class_values: dict mapping column -> list of unique values
        """
        if table_id in self._metadata_cache:
            return self._metadata_cache[table_id]

        df = self._load_table(table_id)
        columns = df.columns.tolist()[: self.max_columns]

        class_values = {}
        for col in columns:
            class_values[col] = df[col].astype(str).unique().tolist()

        result = {
            "columns": columns,
            "class_values": class_values,
        }
        self._metadata_cache[table_id] = result
        return result
