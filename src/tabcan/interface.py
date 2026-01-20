"""
Scikit-learn compatible interface for TabCAN.

Provides fit/sample API for synthetic data generation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import Dataset

from tabcan.model import TabCAN as TabCANModel, create_model
from tabcan.table_registry import TableRegistry
from tabcan.text_embedder import TextEmbedder
from tabcan.vocab_embedder import VocabEmbedder
from tabcan.trainer import train_tabcan, save_checkpoint

_LOG = logging.getLogger(__name__)


class TabCAN(BaseEstimator):
    """
    TabCAN: Tabular Causal Attention Network.

    A simple autoregressive model for tabular data synthesis using
    causal self-attention and LLM embeddings.

    Key features:
    - Autoregressive generation: predicts columns left-to-right
    - LLM embeddings: semantic understanding of values and column names
    - Multi-table support: train on multiple tables, sample from any

    Parameters
    ----------
    model_size : str, default="medium"
        Size of the model ("small", "medium", "large").
    checkpoint : str or None, default=None
        Path to checkpoint for initialization.
    embedding_mode : str, default="text"
        How to compute token embeddings:
        - "text": LLM embeddings via sentence-transformers (semantic, slower)
        - "vocab": Learnable vocabulary embeddings (faster, no semantic transfer)
    max_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=64
        Training batch size.
    learning_rate : float, default=1e-3
        Initial learning rate.
    patience : int, default=20
        Early stopping patience.
    lr_patience : int, default=2
        Reduce LR after this many epochs without improvement.
    lr_factor : float, default=0.8
        Factor to reduce LR by.
    val_split : float, default=0.2
        Validation split fraction.
    shuffle_columns : bool, default=False
        If True, shuffle column order each batch during training.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    registry_ : TableRegistry
        Registry of tables with their schemas.
    model_ : TabCANModel
        Trained TabCAN model.
    train_loss_history_ : list
        Training loss per epoch.
    val_loss_history_ : list
        Validation loss per epoch.
    """

    def __init__(
        self,
        model_size: str = "medium",
        checkpoint: str | None = None,
        embedding_mode: str = "text",
        max_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        patience: int = 20,
        lr_patience: int = 2,
        lr_factor: float = 0.9,
        weight_decay: float = 0.001,
        val_split: float = 0.2,
        shuffle_columns: bool = True,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        if embedding_mode not in ("text", "vocab"):
            raise ValueError(f"embedding_mode must be 'text' or 'vocab', got '{embedding_mode}'")

        self.model_size = model_size
        self.checkpoint = checkpoint
        self.embedding_mode = embedding_mode
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.shuffle_columns = shuffle_columns
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self._fitted = False
        self.registry_: TableRegistry | None = None
        self.model_: TabCANModel | None = None
        # Text mode: single global embedder
        # Vocab mode: per-table embedders
        self.embedder_: TextEmbedder | None = None  # text mode only
        self.embedders_: dict[str, VocabEmbedder] = {}  # vocab mode: table_id -> embedder
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []

    def _get_device(self) -> torch.device:
        """Get the device to use."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_random_state(self):
        """Set random state for reproducibility."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _initialize_model(self, device: torch.device) -> None:
        """Initialize model from checkpoint or create new."""
        if self.checkpoint:
            ckpt = torch.load(self.checkpoint, map_location=device, weights_only=False)
            self.model_ = TabCANModel(ckpt["config"])
            self.model_.load_state_dict(ckpt["model_state_dict"])
            self.model_ = self.model_.to(device)
            if self.verbose > 0:
                _LOG.info(f"Loaded weights from {self.checkpoint}")
        else:
            self.model_ = create_model(self.model_size)
            self.model_ = self.model_.to(device)

    def _create_text_embedder(self, device: torch.device) -> TextEmbedder:
        """Create text embedder for text mode."""
        return TextEmbedder(device=str(device))

    def _create_vocab_embedder(
        self, table_id: str, df: pd.DataFrame, class_values: dict[str, list[str]], device: torch.device
    ) -> VocabEmbedder:
        """Create vocab embedder for a specific table with per-column vocabularies."""
        embedder = VocabEmbedder(device=str(device))

        columns = list(df.columns)
        embedder.build_vocab(columns, class_values)

        if self.verbose > 0:
            total_vocab = sum(embedder.vocab_size(col) for col in columns)
            _LOG.info(
                f"  Table '{table_id}': total_vocab_size={total_vocab}, "
                f"num_columns={embedder.num_columns}"
            )

        return embedder

    def fit(
        self,
        tables: dict[str, pd.DataFrame] | Dataset,
        y=None,
        reset: bool = True,
    ) -> "TabCAN":
        """
        Train on one or more tables.

        Parameters
        ----------
        tables : dict[str, pd.DataFrame] or Dataset
            Either a dict mapping table_id -> DataFrame with categorical columns,
            or a Dataset (e.g., T4Dataset) with get_table_ids() and get_table_metadata().
        y : ignored
        reset : bool, default=True
            If True, start fresh (new model and registry).
            If False, add tables to existing model and train on provided tables.

        Returns
        -------
        self
        """
        self._set_random_state()
        device = self._get_device()

        # Determine if we have a dict of DataFrames or a Dataset
        is_dataset = isinstance(tables, Dataset)

        if is_dataset:
            total_rows = len(tables)
            num_tables = len(tables.get_table_ids())
        else:
            total_rows = sum(len(df) for df in tables.values())
            num_tables = len(tables)

        if reset:
            # Fresh start
            if self.verbose > 0:
                _LOG.info(
                    f"Fitting TabCAN on {num_tables} table(s), {total_rows:,} rows"
                )

            # Create new registry
            self.registry_ = TableRegistry()

            # Initialize model
            self._initialize_model(device)

            # Initialize embedder(s)
            # TextEmbedder is needed for both modes (text: all embeddings, vocab: column names)
            self.embedder_ = self._create_text_embedder(device)
            self.embedders_ = {}  # VocabEmbedders for vocab mode (per-table)
        else:
            # Add to existing model
            if self.model_ is None:
                if self.checkpoint:
                    self._initialize_model(device)
                    # TextEmbedder is needed for both modes
                    self.embedder_ = self._create_text_embedder(device)
                    self.registry_ = TableRegistry()
                else:
                    raise ValueError(
                        "Cannot use reset=False without existing model. "
                        "Either call fit() first or set checkpoint= in constructor."
                    )

            if self.verbose > 0:
                _LOG.info(
                    f"Adding {num_tables} table(s) to existing model, {total_rows:,} rows"
                )

        # Register tables
        if is_dataset:
            # Dataset mode: register from dataset metadata
            for table_id in tables.get_table_ids():
                meta = tables.get_table_metadata(table_id)
                # Normalize table_id to string for consistent handling
                str_table_id = str(table_id)
                self.registry_.register_from_metadata(
                    str_table_id, meta["columns"], meta["class_values"]
                )
                if self.verbose > 0:
                    _LOG.info(
                        f"  Registered table '{str_table_id}': {len(meta['columns'])} columns"
                    )
        else:
            # Dict mode: register DataFrames
            for table_id, df in tables.items():
                df_str = df.astype(str)
                self.registry_.register(table_id, df_str)
                table_info = self.registry_.get(table_id)

                if self.verbose > 0:
                    _LOG.info(
                        f"  Registered table '{table_id}': {len(df)} rows, "
                        f"{len(table_info.columns)} columns"
                    )

        # Convert dict to proper format for trainer
        if not is_dataset:
            tables = {tid: df.astype(str) for tid, df in tables.items()}

        # Compute effective batch size: min of configured batch_size and
        # largest power of 2 <= 10% of dataset size
        max_batch_from_data = total_rows // 10
        if max_batch_from_data >= 2:
            # Largest power of 2 <= max_batch_from_data
            effective_batch_size = 1 << (max_batch_from_data.bit_length() - 1)
            effective_batch_size = min(self.batch_size, effective_batch_size)
        else:
            effective_batch_size = min(self.batch_size, max(1, total_rows))

        if self.verbose > 0 and effective_batch_size != self.batch_size:
            _LOG.info(f"Adjusted batch_size from {self.batch_size} to {effective_batch_size}")

        # Compute effective weight decay: scale inversely with dataset size
        # Reference: 10k rows → use base weight_decay
        # Smaller datasets → higher weight decay (more regularization)
        # Larger datasets → lower weight decay (less regularization)
        reference_size = 10000
        effective_weight_decay = self.weight_decay * (reference_size / total_rows)

        if self.verbose > 0 and effective_weight_decay != self.weight_decay:
            _LOG.info(f"Adjusted weight_decay from {self.weight_decay} to {effective_weight_decay:.6f}")

        # Train using unified trainer
        (
            self.model_,
            self.train_loss_history_,
            self.val_loss_history_,
            vocab_embedders,
        ) = train_tabcan(
            self.model_,
            tables,
            embedder=self.embedder_,
            vocab_embedders=self.embedders_ if self.embedders_ else None,
            embedding_mode=self.embedding_mode,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            batch_size=effective_batch_size,
            patience=self.patience,
            lr_patience=self.lr_patience,
            lr_factor=self.lr_factor,
            weight_decay=effective_weight_decay,
            val_split=self.val_split,
            device=device,
            verbose=self.verbose,
            random_state=self.random_state,
            shuffle_columns=self.shuffle_columns,
        )

        # For vocab mode, capture the embedders created by trainer
        if self.embedding_mode == "vocab" and vocab_embedders:
            self.embedders_ = vocab_embedders

        self._fitted = True
        return self

    def sample(
        self,
        n_samples: int = 1,
        table_id: str | None = None,
        seed_data: pd.DataFrame | None = None,
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples for a specific table.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        table_id : str or None
            Table ID to sample from. Required if multiple tables are registered.
        seed_data : pd.DataFrame or None
            Optional seed data with fixed column values.
        temperature : float, default=1.0
            Sampling temperature (higher = more random).

        Returns
        -------
        pd.DataFrame
            Generated samples with same columns as the specified table.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")

        # Determine table_id
        if table_id is None:
            if len(self.registry_) == 1:
                table_id = next(iter(self.registry_))
            else:
                raise ValueError(
                    f"Multiple tables registered ({list(self.registry_.tables.keys())}). "
                    "Please specify table_id."
                )

        table_info = self.registry_.get(table_id)
        columns = table_info.columns
        class_values = table_info.class_values

        device = self._get_device()
        self.model_.eval()

        # Handle seed data
        if seed_data is not None:
            n_samples = len(seed_data)
            seed_cols = set(seed_data.columns)
        else:
            seed_cols = set()

        # Create local RNG for reproducible sampling
        rng = (
            np.random.RandomState(self.random_state)
            if self.random_state is not None
            else np.random.RandomState()
        )

        # Get the right embedder(s) for this table
        text_embedder = self.embedder_  # Always available for column names
        if self.embedding_mode == "vocab":
            vocab_embedder = self.embedders_[table_id]  # For value embeddings

        # Pre-compute column embeddings (always uses TextEmbedder)
        col_embeddings = text_embedder.embed(columns).to(device)

        # Pre-compute label embeddings for each column
        if self.embedding_mode == "text":
            label_embeddings = {
                col: text_embedder.embed(
                    [str(v) for v in class_values[col]]
                ).to(device)
                for col in columns
            }
        else:
            label_embeddings = {
                col: vocab_embedder.embed_values(
                    col, [str(v) for v in class_values[col]]
                ).to(device)
                for col in columns
            }

        # Initialize output
        generated = {col: [None] * n_samples for col in columns}

        # Copy seed values
        if seed_data is not None:
            for col in seed_cols:
                if col in columns:
                    generated[col] = seed_data[col].astype(str).tolist()

        with torch.no_grad():
            for sample_idx in range(n_samples):
                # Track generated values for this sample
                sample_values = []

                for col_idx, col in enumerate(columns):
                    # Check if this column is seeded
                    if col in seed_cols and generated[col][sample_idx] is not None:
                        sample_values.append(generated[col][sample_idx])
                        continue

                    # Build context from previously generated columns
                    if col_idx == 0:
                        context_emb = torch.empty(
                            1, 0, self.model_.config.embedding_dim, device=device
                        )
                        context_col_emb = torch.empty(
                            1, 0, self.model_.config.embedding_dim, device=device
                        )
                    else:
                        # Get embeddings for previous columns
                        prev_values = sample_values[:col_idx]
                        if self.embedding_mode == "text":
                            context_emb = (
                                text_embedder.embed(prev_values).unsqueeze(0).to(device)
                            )
                        else:
                            # Vocab mode: learnable value embeddings
                            prev_embs = []
                            for prev_col_idx, prev_val in enumerate(prev_values):
                                prev_col = columns[prev_col_idx]
                                emb = vocab_embedder.embed_values(prev_col, [prev_val])
                                prev_embs.append(emb)
                            context_emb = torch.cat(prev_embs, dim=0).unsqueeze(0).to(device)
                        # Both modes use column embeddings
                        context_col_emb = col_embeddings[:col_idx].unsqueeze(0)

                    # Target column embedding
                    target_col_emb = col_embeddings[col_idx].unsqueeze(0)

                    # Forward pass
                    output = self.model_(
                        context_emb,
                        context_col_emb,
                        target_col_emb,
                        is_text_mode=(self.embedding_mode == "text"),
                    )

                    # Compute logits and sample
                    logits = self.model_.compute_logits(
                        output["output"], label_embeddings[col], temperature
                    )
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
                    sampled_idx = rng.choice(len(class_values[col]), p=probs)
                    sampled_value = str(class_values[col][sampled_idx])

                    generated[col][sample_idx] = sampled_value
                    sample_values.append(sampled_value)

        return pd.DataFrame(generated, columns=columns)

    def predict_proba(
        self,
        X: pd.DataFrame,
        target_column: str | None = None,
        table_id: str | None = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Predict class probabilities for target column given features.

        The target column must be the LAST column in the registered table schema.
        All feature columns are used as context.

        Parameters
        ----------
        X : pd.DataFrame
            Features DataFrame. Columns must match the registered schema
            (excluding the target column).
        target_column : str or None
            Column to predict. Default: last column in schema.
        table_id : str or None
            Table ID. Required if multiple tables are registered.
        temperature : float, default=1.0
            Softmax temperature (lower = more confident predictions).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with probability distributions.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Determine table_id
        if table_id is None:
            if len(self.registry_) == 1:
                table_id = next(iter(self.registry_))
            else:
                raise ValueError(
                    f"Multiple tables registered ({list(self.registry_.tables.keys())}). "
                    "Please specify table_id."
                )

        table_info = self.registry_.get(table_id)
        columns = table_info.columns
        class_values = table_info.class_values

        # Default to last column as target
        if target_column is None:
            target_column = columns[-1]

        if target_column not in columns:
            raise ValueError(f"Target column '{target_column}' not in table schema: {columns}")

        # Target must be the last column for autoregressive prediction
        target_idx = columns.index(target_column)
        if target_idx != len(columns) - 1:
            raise ValueError(
                f"Target column '{target_column}' must be the last column in the schema. "
                f"Current order: {columns}. Consider reordering columns when calling fit()."
            )

        # Feature columns are all columns except target
        feature_columns = columns[:-1]

        # Validate X has all feature columns
        missing_cols = set(feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns in X: {missing_cols}")

        device = self._get_device()
        self.model_.eval()

        n_samples = len(X)

        # Get the right embedder(s) for this table
        text_embedder = self.embedder_
        if self.embedding_mode == "vocab":
            vocab_embedder = self.embedders_[table_id]

        # Pre-compute column embeddings
        col_embeddings = text_embedder.embed(columns).to(device)

        # Pre-compute label embeddings for target column
        target_classes = class_values[target_column]
        if self.embedding_mode == "text":
            label_emb = text_embedder.embed([str(v) for v in target_classes]).to(device)
        else:
            label_emb = vocab_embedder.embed_values(
                target_column, [str(v) for v in target_classes]
            ).to(device)

        # Process all samples
        all_probs = []

        with torch.no_grad():
            for sample_idx in range(n_samples):
                # Get feature values for this sample (in column order)
                feature_values = [str(X[col].iloc[sample_idx]) for col in feature_columns]

                # Build context embeddings
                if self.embedding_mode == "text":
                    context_emb = text_embedder.embed(feature_values).unsqueeze(0).to(device)
                else:
                    context_embs = []
                    for col_idx, val in enumerate(feature_values):
                        col = feature_columns[col_idx]
                        emb = vocab_embedder.embed_values(col, [val])
                        context_embs.append(emb)
                    context_emb = torch.cat(context_embs, dim=0).unsqueeze(0).to(device)

                # Context column embeddings (all except target)
                context_col_emb = col_embeddings[:-1].unsqueeze(0)

                # Target column embedding
                target_col_emb = col_embeddings[-1].unsqueeze(0)

                # Forward pass
                output = self.model_(
                    context_emb,
                    context_col_emb,
                    target_col_emb,
                    is_text_mode=(self.embedding_mode == "text"),
                )

                # Compute probabilities
                logits = self.model_.compute_logits(output["output"], label_emb, temperature)
                probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
                all_probs.append(probs)

        return np.array(all_probs)

    def predict(
        self,
        X: pd.DataFrame,
        target_column: str | None = None,
        table_id: str | None = None,
    ) -> np.ndarray:
        """
        Predict class labels for target column given features.

        Parameters
        ----------
        X : pd.DataFrame
            Features DataFrame.
        target_column : str or None
            Column to predict. Default: last column in schema.
        table_id : str or None
            Table ID. Required if multiple tables are registered.

        Returns
        -------
        np.ndarray
            Array of predicted class labels.
        """
        probs = self.predict_proba(X, target_column, table_id)

        # Get table info for class labels
        if table_id is None:
            table_id = next(iter(self.registry_))
        table_info = self.registry_.get(table_id)
        if target_column is None:
            target_column = table_info.columns[-1]

        classes = table_info.class_values[target_column]
        return np.array(classes)[np.argmax(probs, axis=1)]

    def get_classes(
        self,
        target_column: str | None = None,
        table_id: str | None = None,
    ) -> list[str]:
        """
        Get class labels for target column.

        Parameters
        ----------
        target_column : str or None
            Target column. Default: last column in schema.
        table_id : str or None
            Table ID. Required if multiple tables are registered.

        Returns
        -------
        list[str]
            List of class labels in order.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first.")

        if table_id is None:
            if len(self.registry_) == 1:
                table_id = next(iter(self.registry_))
            else:
                raise ValueError("Multiple tables registered. Please specify table_id.")

        table_info = self.registry_.get(table_id)
        if target_column is None:
            target_column = table_info.columns[-1]

        return table_info.class_values[target_column]

    def save(self, path: str | Path) -> None:
        """
        Save the fitted model to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the model checkpoint.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before saving.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model_.state_dict(),
            "config": self.model_.config,
            "registry": self.registry_.state_dict(),
            "params": {
                "model_size": self.model_size,
                "embedding_mode": self.embedding_mode,
            },
        }

        # Save embedder(s)
        if self.embedding_mode == "text" and self.embedder_ is not None:
            checkpoint["embedder_state"] = self.embedder_.state_dict()
        elif self.embedding_mode == "vocab" and self.embedders_:
            # Save per-table embedders
            checkpoint["embedders_state"] = {
                table_id: emb.state_dict()
                for table_id, emb in self.embedders_.items()
            }

        torch.save(checkpoint, path)
        _LOG.info(f"Saved TabCAN model to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TabCAN":
        """
        Load model from disk.

        Parameters
        ----------
        path : str or Path
            Path to saved model.
        device : str or None
            Device to load model on.

        Returns
        -------
        TabCAN
            Loaded model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Create instance
        params = checkpoint.get("params", {})
        instance = cls(**params)

        # Load registry
        instance.registry_ = TableRegistry.load_state_dict(checkpoint["registry"])

        # Load model
        instance.model_ = TabCANModel(checkpoint["config"])
        instance.model_.load_state_dict(checkpoint["model_state_dict"])
        instance.model_ = instance.model_.to(device)
        instance.model_.eval()

        # Load embedder(s) based on embedding_mode
        # TextEmbedder is always needed (for column embeddings in both modes)
        instance.embedder_ = TextEmbedder(device=device)
        instance.embedders_ = {}

        embedding_mode = params.get("embedding_mode", "text")
        if embedding_mode == "vocab":
            # Load per-table VocabEmbedders
            if "embedders_state" in checkpoint:
                for table_id, state in checkpoint["embedders_state"].items():
                    emb = VocabEmbedder.from_state_dict(state, device=device)
                    instance.embedders_[table_id] = emb
        else:
            # Load TextEmbedder state
            if "embedder_state" in checkpoint:
                instance.embedder_.load_state_dict(checkpoint["embedder_state"])
            elif "text_embedder_state" in checkpoint:
                # Backwards compatibility with old checkpoints
                instance.embedder_.load_state_dict(checkpoint["text_embedder_state"])

        instance._fitted = True

        _LOG.info(f"Loaded TabCAN model from {path}")
        return instance
