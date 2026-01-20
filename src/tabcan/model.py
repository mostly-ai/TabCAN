"""
TabCAN Model: Tabular Causal Attention Network.

A simple autoregressive model for tabular data using causal self-attention.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabcan.text_embedder import EMBEDDING_DIM


@dataclass
class TabCANConfig:
    """Configuration for TabCAN model."""

    embedding_dim: int = EMBEDDING_DIM  # LLM embedding dimension (384 for MiniLM)
    hidden_size: int = 256
    num_heads: int = 8
    dropout: float = 0.2
    max_columns: int = 64  # Maximum number of columns for causal mask


# Predefined model sizes
MODEL_SIZES = {
    "small": TabCANConfig(
        embedding_dim=EMBEDDING_DIM,
        hidden_size=128,
        num_heads=4,
    ),
    "medium": TabCANConfig(
        embedding_dim=EMBEDDING_DIM,
        hidden_size=256,
        num_heads=8,
    ),
    "large": TabCANConfig(
        embedding_dim=EMBEDDING_DIM,
        hidden_size=512,
        num_heads=8,
    ),
}


def create_model(size: str = "medium", **kwargs) -> "TabCAN":
    """
    Create a TabCAN model with predefined size.

    Args:
        size: One of "small", "medium", "large"
        **kwargs: Override config values

    Returns:
        Initialized model
    """
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown size: {size}. Choose from {list(MODEL_SIZES.keys())}")

    config = MODEL_SIZES[size]

    # Apply overrides
    if kwargs:
        config = TabCANConfig(
            embedding_dim=kwargs.get("embedding_dim", config.embedding_dim),
            hidden_size=kwargs.get("hidden_size", config.hidden_size),
            num_heads=kwargs.get("num_heads", config.num_heads),
            dropout=kwargs.get("dropout", config.dropout),
        )

    return TabCAN(config)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer for autoregressive modeling.

    Each position can only attend to previous positions (and itself).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden) - input sequence
            attn_mask: (seq_len, seq_len) - True for positions to mask (not attend)

        Returns:
            (batch, seq_len, hidden) - attended sequence
        """
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + out)  # Residual connection + LayerNorm


class TabCAN(nn.Module):
    """
    TabCAN: Tabular Causal Attention Network.

    Predicts column values autoregressively using causal self-attention
    and cosine similarity to candidate embeddings.
    """

    def __init__(self, config: TabCANConfig):
        super().__init__()
        self.config = config

        # Start token embedding for predicting first column
        self.start_emb = nn.Parameter(torch.zeros(1, config.hidden_size))

        # Projections for input embeddings (embedding_dim -> hidden_size)
        self.value_proj = nn.Linear(config.embedding_dim, config.hidden_size)
        self.column_proj = nn.Linear(config.embedding_dim, config.hidden_size)
        self.target_proj = nn.Linear(config.embedding_dim, config.hidden_size)

        # Causal self-attention
        self.attn = CausalSelfAttention(
            config.hidden_size, config.num_heads, config.dropout
        )

        # Input normalization
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Output projection (hidden*2 -> embedding_dim for cosine similarity)
        self.output_proj = nn.Linear(config.hidden_size * 2, config.embedding_dim)

        # Causal mask buffer (upper triangular = positions to mask)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_columns, config.max_columns), diagonal=1
            ).bool(),
            persistent=False,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        nn.init.normal_(self.start_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.column_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.column_proj.bias)
        nn.init.normal_(self.target_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.target_proj.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        context_emb: torch.Tensor,
        context_col_emb: torch.Tensor,
        target_col_emb: torch.Tensor,
        is_text_mode: bool = True,
    ) -> dict:
        """
        Forward pass for inference (single column prediction).

        Args:
            context_emb: (batch, context_len, 384) - context value embeddings
            context_col_emb: (batch, context_len, 384) - context column embeddings
            target_col_emb: (batch, 384) - target column embedding
            is_text_mode: True for text mode, False for vocab mode (unused, kept for API)

        Returns:
            Dict with 'output' (batch, 384)
        """
        batch_size = target_col_emb.size(0)
        context_len = context_emb.size(1)

        if context_len == 0:
            # First column: just START token
            h = self.start_emb.expand(batch_size, 1, -1)  # (batch, 1, hidden)
        else:
            # Build sequence: [START, val0+col0, val1+col1, ...]
            start_h = self.start_emb.expand(batch_size, 1, -1)  # (batch, 1, hidden)
            context_h = self.value_proj(context_emb) + self.column_proj(context_col_emb)
            h = torch.cat([start_h, context_h], dim=1)  # (batch, context_len+1, hidden)

        h = self.norm(self.dropout(h))

        # Causal attention
        seq_len = h.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        h = self.attn(h, attn_mask=causal_mask)

        # Get attended representation from last position
        attended = h[:, -1, :]  # (batch, hidden)

        # Concatenate with target column embedding
        target_h = self.target_proj(target_col_emb)  # (batch, hidden)
        combined = torch.cat([attended, target_h], dim=-1)  # (batch, hidden*2)

        # Output projection
        output = self.output_proj(combined)  # (batch, embedding_dim)

        return {"output": output}

    def compute_loss_all_columns(
        self,
        value_emb: torch.Tensor,
        col_emb: torch.Tensor,
        label_embeddings: dict[str, torch.Tensor],
        value_to_idx: dict[str, dict[str, int]],
        columns: list[str],
        values: list[list[str]],
        is_text_mode: bool = True,
    ) -> torch.Tensor:
        """
        Compute loss for all columns in a single forward pass using shifted targets.

        Input sequence:  [START, val0+col0, val1+col1, ..., val_{n-2}+col_{n-2}]
        Output targets:  [val0,  val1,      val2,      ...,  val_{n-1}]

        Args:
            value_emb: (batch, num_cols, 384) - value embeddings for all columns
            col_emb: (num_cols, 384) - column embeddings
            label_embeddings: dict mapping column name -> (num_classes, 384)
            value_to_idx: dict mapping column name -> {value: index}
            columns: list of column names
            values: list of lists of string values, (batch, num_cols)
            is_text_mode: True for text mode, False for vocab mode (unused, kept for API)

        Returns:
            Scalar loss tensor
        """
        batch_size = value_emb.size(0)
        num_cols = len(columns)
        device = value_emb.device

        # Build input sequence with shifted targets
        # Position 0: START (predicts val_0)
        # Position i (i>0): val_{i-1} + col_{i-1} (predicts val_i)

        # Start token for position 0
        start_h = self.start_emb.expand(batch_size, 1, -1)  # (batch, 1, hidden)

        # For positions 1..n-1: project value + column from previous position
        value_h = self.value_proj(value_emb[:, :-1, :])  # (batch, num_cols-1, hidden)
        col_h = self.column_proj(col_emb[:-1]).unsqueeze(0).expand(batch_size, -1, -1)
        rest = value_h + col_h  # (batch, num_cols-1, hidden)

        h = torch.cat([start_h, rest], dim=1)  # (batch, num_cols, hidden)

        h = self.norm(self.dropout(h))

        # Causal self-attention
        causal_mask = self.causal_mask[:num_cols, :num_cols]
        h = self.attn(h, attn_mask=causal_mask)  # (batch, num_cols, hidden)

        # Project target columns
        target_h = self.target_proj(col_emb)  # (num_cols, hidden)
        target_h = target_h.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_cols, hidden)

        # Concatenate attended hidden with target column for each position
        combined = torch.cat([h, target_h], dim=-1)  # (batch, num_cols, hidden*2)

        # Project to embedding space
        outputs = self.output_proj(combined)  # (batch, num_cols, embedding_dim)

        # Compute loss for each column
        total_loss = torch.tensor(0.0, device=device)
        total_count = 0

        for col_idx, col in enumerate(columns):
            # Get labels for this column
            label_indices = [value_to_idx[col].get(values[i][col_idx], -1) for i in range(batch_size)]
            valid_mask = [idx >= 0 for idx in label_indices]

            if not any(valid_mask):
                continue

            valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
            valid_labels = torch.tensor(
                [label_indices[i] for i in valid_indices], device=device
            )

            # Get output for this column position
            col_output = outputs[valid_indices, col_idx, :]  # (num_valid, embedding_dim)

            # Compute logits and loss
            logits = self.compute_logits(col_output, label_embeddings[col])
            loss = F.cross_entropy(logits, valid_labels, reduction="sum")
            total_loss = total_loss + loss
            total_count += len(valid_indices)

        if total_count > 0:
            return total_loss / total_count
        return total_loss

    def compute_logits(
        self,
        output: torch.Tensor,
        candidate_emb: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute logits via cosine similarity.

        Args:
            output: Model output embeddings, shape (..., embedding_dim)
            candidate_emb: Embeddings of candidate values, shape (num_candidates, embedding_dim)
            temperature: Softmax temperature. Lower = sharper distribution.

        Returns:
            Logits, shape (..., num_candidates)
        """
        output_norm = F.normalize(output, dim=-1)
        candidate_norm = F.normalize(candidate_emb, dim=-1)
        # Scale by 20 (typical for contrastive learning) / temperature
        return torch.matmul(output_norm, candidate_norm.t()) * (20.0 / temperature)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
