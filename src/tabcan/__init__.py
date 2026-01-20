"""TabCAN - Tabular Causal Attention Network."""

from tabcan.interface import TabCAN
from tabcan.model import (
    TabCAN as TabCANModel,
    TabCANConfig,
    CausalSelfAttention,
    create_model,
    MODEL_SIZES,
)
from tabcan.text_embedder import TextEmbedder, EMBEDDING_DIM
from tabcan.vocab_embedder import VocabEmbedder
from tabcan.trainer import (
    train_tabcan,
    save_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    # High-level interface
    "TabCAN",
    # Model
    "TabCANModel",
    "TabCANConfig",
    "CausalSelfAttention",
    "create_model",
    "MODEL_SIZES",
    # Embedders
    "TextEmbedder",
    "VocabEmbedder",
    "EMBEDDING_DIM",
    # Training
    "train_tabcan",
    "save_checkpoint",
]
