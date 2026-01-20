"""
TableRegistry: Registry for managing multiple tables with their schemas.

Each table stores its column order and unique class values.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TableInfo:
    """Metadata for a registered table."""

    table_id: str
    columns: list[str]
    class_values: dict[str, list[str]]  # col -> unique values


class TableRegistry:
    """
    Registry of tables for multi-table support.

    Each table is registered with:
    - Its schema (column names and unique values)
    """

    def __init__(self):
        self.tables: dict[str, TableInfo] = {}

    def register(
        self,
        table_id: str,
        df: pd.DataFrame,
    ) -> TableInfo:
        """
        Register a new table.

        Args:
            table_id: Unique identifier for the table
            df: DataFrame containing the table data (all string values)

        Returns:
            TableInfo for the registered table
        """
        columns = list(df.columns)
        class_values = {
            col: sorted(df[col].astype(str).unique().tolist()) for col in columns
        }

        info = TableInfo(table_id, columns, class_values)
        self.tables[table_id] = info
        return info

    def register_from_metadata(
        self,
        table_id: str,
        columns: list[str],
        class_values: dict[str, list[str]],
    ) -> TableInfo:
        """
        Register a table from pre-computed metadata.

        Args:
            table_id: Unique identifier for the table
            columns: List of column names
            class_values: Dict mapping column -> unique values

        Returns:
            TableInfo for the registered table
        """
        info = TableInfo(table_id, columns, class_values)
        self.tables[table_id] = info
        return info

    def get(self, table_id: str) -> TableInfo:
        """Get table info by ID."""
        if table_id not in self.tables:
            raise KeyError(f"Table '{table_id}' not found in registry")
        return self.tables[table_id]

    def __contains__(self, table_id: str) -> bool:
        """Check if table is registered."""
        return table_id in self.tables

    def __len__(self) -> int:
        """Number of registered tables."""
        return len(self.tables)

    def __iter__(self):
        """Iterate over table IDs."""
        return iter(self.tables)

    def state_dict(self) -> dict:
        """
        Serialize for checkpoint.

        Returns:
            Dict containing all table metadata
        """
        return {
            tid: {
                "columns": info.columns,
                "class_values": info.class_values,
            }
            for tid, info in self.tables.items()
        }

    @classmethod
    def load_state_dict(cls, state: dict) -> "TableRegistry":
        """
        Deserialize from checkpoint.

        Args:
            state: Dict from state_dict()

        Returns:
            Restored TableRegistry
        """
        registry = cls()
        for tid, data in state.items():
            info = TableInfo(
                table_id=tid,
                columns=data["columns"],
                class_values=data["class_values"],
            )
            registry.tables[tid] = info
        return registry
