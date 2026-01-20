"""
Dataset utilities for TabCAN training.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiTableDataset(Dataset):
    """
    Dataset for training on multiple tables with potentially different schemas.

    Supports both single-table and multi-table training. Each item includes
    the table_id so the trainer can handle variable schemas.

    Args:
        tables: Dict mapping table_id -> DataFrame

    Example:
        >>> dataset = MultiTableDataset({"products": df1, "customers": df2})
        >>> item = dataset[0]
        >>> print(item["table_id"])   # 'products'
        >>> print(item["values"])     # ['red', 'S', 'low']
        >>> print(item["columns"])    # ['color', 'size', 'price']
    """

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables
        self.table_ids = list(tables.keys())

        # Build global row index: [(table_id, row_idx), ...]
        self.row_index: list[tuple[str, int]] = []
        for table_id, df in tables.items():
            for row_idx in range(len(df)):
                self.row_index.append((table_id, row_idx))

        # Cache table data as string lists
        self._table_data: dict[str, list[list[str]]] = {}
        self._table_columns: dict[str, list[str]] = {}
        self._class_values: dict[str, dict[str, list[str]]] = {}
        for table_id, df in tables.items():
            self._table_data[table_id] = df.astype(str).values.tolist()
            self._table_columns[table_id] = df.columns.tolist()
            self._class_values[table_id] = {
                col: df[col].astype(str).unique().tolist() for col in df.columns
            }

    def __len__(self) -> int:
        return len(self.row_index)

    def __getitem__(self, idx: int) -> dict:
        table_id, row_idx = self.row_index[idx]
        return {
            "table_id": table_id,
            "values": self._table_data[table_id][row_idx],
            "columns": self._table_columns[table_id],
        }

    def get_table_ids(self) -> list[str]:
        """Get list of unique table IDs."""
        return self.table_ids

    def get_table_columns(self, table_id: str) -> list[str]:
        """Get columns for a specific table."""
        return self._table_columns[table_id]

    def get_class_values(self, table_id: str) -> dict[str, list[str]]:
        """Get unique values for each column of a table."""
        return self._class_values[table_id]


def collate_multitable(batch: list[dict]) -> dict:
    """
    Collate function for MultiTableDataset and T4Dataset.

    Pads all rows to the maximum number of columns in the batch.

    Args:
        batch: List of dicts with keys:
            - "values" or "text_values": List[str] values
            - "columns": List[str] column names
            - "table_id": str or int table ID

    Returns:
        Dict with:
        - values: List[List[str]] padded to max_cols
        - columns: List[List[str]] per-sample column names padded to max_cols
        - table_ids: List[str] table ID per row
        - attention_mask: Tensor (batch, max_cols) - True for valid positions
    """
    max_cols = max(len(b["columns"]) for b in batch)

    values = []
    columns_list = []
    table_ids = []
    attention_mask = []

    for item in batch:
        n = len(item["columns"])
        # Handle both "values" (MultiTableDataset) and "text_values" (T4Dataset)
        item_values = item.get("values") or item.get("text_values")
        # Pad values with empty strings
        values.append(item_values + [""] * (max_cols - n))
        # Pad columns with empty strings
        columns_list.append(item["columns"] + [""] * (max_cols - n))
        # Normalize table_id to string
        table_ids.append(str(item["table_id"]))
        # Mask: True for valid, False for padding
        attention_mask.append([True] * n + [False] * (max_cols - n))

    return {
        "values": values,
        "columns": columns_list,
        "table_ids": table_ids,
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }


# Keep old names for backwards compatibility
class TabCANDataset(Dataset):
    """
    Single-table dataset (legacy, wraps MultiTableDataset).

    Args:
        df: DataFrame with categorical columns
    """

    def __init__(self, df: pd.DataFrame):
        self._multi = MultiTableDataset({"default": df})
        self.columns = df.columns.tolist()
        self.data = df.astype(str).values.tolist()

    def __len__(self) -> int:
        return len(self._multi)

    def __getitem__(self, idx: int) -> dict:
        item = self._multi[idx]
        return {
            "values": item["values"],
            "columns": item["columns"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Legacy collate function for TabCANDataset."""
    return {
        "values": [b["values"] for b in batch],
        "columns": batch[0]["columns"],
    }
