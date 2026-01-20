"""
Vocabulary-based embedder using learnable nn.Embedding.

Alternative to TextEmbedder for faster training without LLM inference.
Each column has its own value vocabulary + learned column embeddings.
"""

import torch
import torch.nn as nn


class VocabEmbedder(nn.Module):
    """
    Per-table embedder with separate vocabulary for each column.

    - Value embeddings: each column has its own nn.Embedding for unique values
    - Column embeddings: learned nn.Embedding (not text-based)

    Args:
        embedding_dim: Dimension of embeddings. Defaults to 384 (same as MiniLM).
        device: Device to run embeddings on.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        device: str | None = None,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Column info
        self._columns: list[str] = []
        self._col_to_idx: dict[str, int] = {}

        # Per-column value vocabulary: col_name -> {value: idx}
        self._vocabs: dict[str, dict[str, int]] = {}
        self._vocab_lists: dict[str, list[str]] = {}  # col_name -> [values]

        # Per-column value embeddings: col_name -> nn.Embedding
        self._value_embeddings: nn.ModuleDict = nn.ModuleDict()

        # Learned column embeddings (not text-based)
        self._column_embedding: nn.Embedding | None = None

        self._frozen = False

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    @property
    def num_columns(self) -> int:
        """Return the number of columns."""
        return len(self._columns)

    def _safe_key(self, col: str) -> str:
        """Generate safe key for nn.ModuleDict (avoids reserved attribute conflicts)."""
        return "col_" + col.replace(".", "_").replace("-", "_").replace(" ", "_")

    def vocab_size(self, column: str) -> int:
        """Return vocabulary size for a column."""
        return len(self._vocabs.get(column, {}))

    def build_vocab(self, columns: list[str], class_values: dict[str, list[str]]) -> None:
        """
        Build per-column vocabularies and embeddings.

        Args:
            columns: List of column names (order matters).
            class_values: Dict mapping column_name -> list of unique values.
        """
        if self._frozen:
            raise RuntimeError("Cannot build vocab after embedder is frozen")

        self._columns = list(columns)
        self._col_to_idx = {col: i for i, col in enumerate(columns)}

        # Build per-column vocabularies and embeddings
        for col in columns:
            values = class_values[col]
            # Build vocab: value -> index (0 is UNK)
            vocab = {"<UNK>": 0}
            vocab_list = ["<UNK>"]
            for val in values:
                val_str = str(val)
                if val_str not in vocab:
                    vocab[val_str] = len(vocab)
                    vocab_list.append(val_str)

            self._vocabs[col] = vocab
            self._vocab_lists[col] = vocab_list

            # Create embedding for this column
            emb = nn.Embedding(len(vocab), self._embedding_dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self._value_embeddings[self._safe_key(col)] = emb

        # Create column embeddings (learned, not text-based)
        self._column_embedding = nn.Embedding(len(columns), self._embedding_dim)
        nn.init.normal_(self._column_embedding.weight, mean=0.0, std=0.02)

        # Move to device
        self.to(self.device)

    def freeze(self) -> None:
        """Freeze the vocabulary (no more additions allowed)."""
        self._frozen = True

    def embed(self, texts: list[str], show_progress: bool = False) -> torch.Tensor:
        """
        Embed a list of values (for API compatibility).

        NOTE: This is a fallback that uses a simple hash.
        For proper per-column embedding, use embed_values().

        Args:
            texts: List of value strings to embed.
            show_progress: Ignored.

        Returns:
            Tensor of shape (len(texts), embedding_dim).
        """
        # This is a fallback - tries to find values in any column vocab
        if not texts:
            return torch.empty(0, self._embedding_dim, device=self.device)

        results = []
        for text in texts:
            text_str = str(text)
            found = False
            # Search through all columns for this value
            for col in self._columns:
                if text_str in self._vocabs[col]:
                    idx = self._vocabs[col][text_str]
                    emb = self._value_embeddings[self._safe_key(col)](
                        torch.tensor([idx], device=self.device)
                    )
                    results.append(emb.squeeze(0))
                    found = True
                    break
            if not found:
                # Return zeros for unknown
                results.append(torch.zeros(self._embedding_dim, device=self.device))

        return torch.stack(results)

    def embed_values(self, column: str, values: list[str]) -> torch.Tensor:
        """
        Embed values for a specific column.

        Args:
            column: Column name.
            values: List of values to embed.

        Returns:
            Tensor of shape (len(values), embedding_dim).
        """
        if not values:
            return torch.empty(0, self._embedding_dim, device=self.device)

        if column not in self._vocabs:
            raise KeyError(f"Column '{column}' not in vocabulary")

        vocab = self._vocabs[column]
        indices = [vocab.get(str(v), 0) for v in values]  # 0 is UNK
        indices_tensor = torch.tensor(indices, device=self.device)

        return self._value_embeddings[self._safe_key(column)](indices_tensor)

    def embed_columns(self, columns: list[str] | None = None) -> torch.Tensor:
        """
        Get learned column embeddings.

        Args:
            columns: List of column names. If None, returns all columns in order.

        Returns:
            Tensor of shape (len(columns), embedding_dim).
        """
        if columns is None:
            columns = self._columns

        if not columns:
            return torch.empty(0, self._embedding_dim, device=self.device)

        if self._column_embedding is None:
            raise RuntimeError("Must call build_vocab() before embed_columns()")

        indices = [self._col_to_idx.get(col, 0) for col in columns]
        indices_tensor = torch.tensor(indices, device=self.device)
        return self._column_embedding(indices_tensor)

    def embed_single(self, text: str) -> torch.Tensor:
        """Embed a single value (fallback)."""
        return self.embed([text])[0]

    def precompute(self, texts: list[str], show_progress: bool = True) -> None:
        """No-op for API compatibility."""
        pass

    def clear_cache(self) -> None:
        """No-op for API compatibility."""
        pass

    def to(self, device: str) -> "VocabEmbedder":
        """Move embedder to device."""
        self.device = device
        for emb in self._value_embeddings.values():
            emb.to(device)
        if self._column_embedding is not None:
            self._column_embedding = self._column_embedding.to(device)
        return self

    def parameters(self):
        """Return all embedding parameters for optimizer."""
        params = []
        for emb in self._value_embeddings.values():
            params.extend(emb.parameters())
        if self._column_embedding is not None:
            params.extend(self._column_embedding.parameters())
        return iter(params)

    def train(self, mode: bool = True):
        """Set training mode."""
        for emb in self._value_embeddings.values():
            emb.train(mode)
        if self._column_embedding is not None:
            self._column_embedding.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def state_dict(self) -> dict:
        """Get state for serialization."""
        state = {
            "embedding_dim": self._embedding_dim,
            "columns": self._columns,
            "col_to_idx": self._col_to_idx,
            "vocabs": self._vocabs,
            "vocab_lists": self._vocab_lists,
            "frozen": self._frozen,
            "value_embeddings": {
                k: v.weight.data.cpu() for k, v in self._value_embeddings.items()
            },
        }
        if self._column_embedding is not None:
            state["column_embedding_weight"] = self._column_embedding.weight.data.cpu()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load state from serialization."""
        self._embedding_dim = state["embedding_dim"]
        self._columns = state["columns"]
        self._col_to_idx = state["col_to_idx"]
        self._vocabs = state["vocabs"]
        self._vocab_lists = state["vocab_lists"]
        self._frozen = state.get("frozen", False)

        # Restore value embeddings
        self._value_embeddings = nn.ModuleDict()
        for key, weight in state["value_embeddings"].items():
            emb = nn.Embedding(weight.size(0), self._embedding_dim)
            emb.weight.data = weight.to(self.device)
            self._value_embeddings[key] = emb

        # Restore column embedding
        if "column_embedding_weight" in state:
            weight = state["column_embedding_weight"]
            self._column_embedding = nn.Embedding(weight.size(0), self._embedding_dim)
            self._column_embedding.weight.data = weight.to(self.device)

    @classmethod
    def from_state_dict(cls, state: dict, device: str | None = None) -> "VocabEmbedder":
        """Create embedder from state dict."""
        embedder = cls(embedding_dim=state["embedding_dim"], device=device)
        embedder.load_state_dict(state)
        return embedder
