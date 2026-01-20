"""
Training utilities for TabCAN.

Unified trainer for single-table and multi-table training.
"""

import copy
import logging
import random

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from tabcan.dataset import MultiTableDataset, collate_multitable
from tabcan.model import TabCAN
from tabcan.text_embedder import TextEmbedder
from tabcan.vocab_embedder import VocabEmbedder

_LOG = logging.getLogger(__name__)


def train_tabcan(
    model: TabCAN,
    tables: dict,
    embedder: TextEmbedder | None = None,
    vocab_embedders: dict[str, VocabEmbedder] | None = None,
    *,
    embedding_mode: str = "text",
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    patience: int = 20,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    weight_decay: float = 0.01,
    val_split: float = 0.2,
    device: str | torch.device | None = None,
    verbose: int = 0,
    min_delta: float = 1e-4,
    min_lr: float = 1e-6,
    num_workers: int = 0,
    random_state: int | None = None,
    shuffle_columns: bool = False,
) -> tuple[TabCAN, list[float], list[float], dict[str, VocabEmbedder] | None]:
    """
    Train TabCAN model on one or more tables.

    Handles both single-table and multi-table training uniformly.
    Uses efficient single forward pass for all columns.

    Args:
        model: TabCAN model to train
        tables: Dict mapping table_id -> DataFrame, or MultiTableDataset/T4Dataset
        embedder: TextEmbedder instance (required for text mode)
        vocab_embedders: Dict mapping table_id -> VocabEmbedder (for vocab mode)
        embedding_mode: "text" or "vocab"
        max_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size
        patience: Early stopping patience
        lr_patience: Reduce LR patience
        lr_factor: LR reduction factor
        weight_decay: L2 regularization
        val_split: Validation split fraction
        device: Device to train on
        verbose: Verbosity level
        min_delta: Minimum improvement for early stopping
        min_lr: Minimum learning rate
        num_workers: DataLoader workers
        random_state: Random seed
        shuffle_columns: If True, shuffle column order each batch during training

    Returns:
        Tuple of (best model, train_loss_history, val_loss_history, vocab_embedders or None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    is_vocab_mode = embedding_mode == "vocab"

    model = model.to(device)

    # TextEmbedder is needed for both modes (text mode: all embeddings, vocab mode: column names)
    if embedder is None:
        embedder = TextEmbedder(device=str(device))
    else:
        embedder = embedder.to(device)

    # Handle different dataset types
    if isinstance(tables, dict):
        dataset = MultiTableDataset(tables)
    else:
        # Assume it's already a dataset (e.g., T4Dataset)
        dataset = tables

    # For vocab mode, pre-create all VocabEmbedders before training
    # so their parameters are included in the optimizer
    if is_vocab_mode:
        if vocab_embedders is None:
            vocab_embedders = {}

        # Get all unique table IDs from dataset
        if hasattr(dataset, 'get_table_ids'):
            all_table_ids = [str(tid) for tid in dataset.get_table_ids()]
        else:
            all_table_ids = list(dataset.table_ids)

        embed_dim = model.config.embedding_dim
        _LOG.info(f"Pre-creating VocabEmbedders for {len(all_table_ids)} tables...")

        for table_id in all_table_ids:
            if table_id not in vocab_embedders:
                # Get table metadata
                if hasattr(dataset, 'get_table_metadata'):
                    meta = dataset.get_table_metadata(table_id)
                    columns = meta["columns"]
                    class_values = meta["class_values"]
                else:
                    columns = dataset.get_table_columns(table_id)
                    class_values = dataset.get_class_values(table_id)

                emb = VocabEmbedder(embedding_dim=embed_dim, device=str(device))
                emb.build_vocab(columns, class_values)
                vocab_embedders[table_id] = emb

        _LOG.info(f"Created {len(vocab_embedders)} VocabEmbedders")

    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator()
    if random_state is not None:
        generator.manual_seed(random_state)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    _LOG.info(f"Dataset split: {train_size:,} train, {val_size:,} validation rows")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multitable,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multitable,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Collect parameters to optimize
    params_to_optimize = list(model.parameters())
    if is_vocab_mode and vocab_embedders:
        for emb in vocab_embedders.values():
            params_to_optimize.extend(emb.parameters())

    optimizer = AdamW(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    # Early stopping state
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    train_loss_history = []
    val_loss_history = []

    # For column shuffling
    col_rng = random.Random(random_state)

    try:
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            if is_vocab_mode and vocab_embedders:
                for emb in vocab_embedders.values():
                    emb.train()

            train_loss = 0.0
            train_predictions = 0

            for batch_idx, batch in enumerate(train_loader):
                batch_loss, num_preds = _process_batch(
                    model, batch, embedder, device, dataset,
                    training=True,
                    is_vocab_mode=is_vocab_mode,
                    vocab_embedders=vocab_embedders,
                    shuffle_columns=shuffle_columns,
                    col_rng=col_rng,
                )

                if num_preds > 0:
                    avg_batch_loss = batch_loss / num_preds
                    optimizer.zero_grad()
                    avg_batch_loss.backward()
                    optimizer.step()

                    train_loss += batch_loss.item()
                    train_predictions += num_preds

                if verbose > 0 and (batch_idx + 1) % 100 == 0:
                    _LOG.info(
                        f"  Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {batch_loss.item() / max(num_preds, 1):.4f}"
                    )

            avg_train_loss = (
                train_loss / train_predictions if train_predictions > 0 else 0
            )
            train_loss_history.append(avg_train_loss)

            # Validation phase
            model.eval()
            if is_vocab_mode and vocab_embedders:
                for emb in vocab_embedders.values():
                    emb.eval()

            val_loss = 0.0
            val_predictions = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch_loss, num_preds = _process_batch(
                        model, batch, embedder, device, dataset,
                        training=False,
                        is_vocab_mode=is_vocab_mode,
                        vocab_embedders=vocab_embedders,
                        shuffle_columns=False,  # No shuffling for validation
                        col_rng=None,
                    )
                    val_loss += batch_loss.item()
                    val_predictions += num_preds

            avg_val_loss = val_loss / val_predictions if val_predictions > 0 else 0
            val_loss_history.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if verbose > 0:
                _LOG.info(
                    f"Epoch {epoch + 1}/{max_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"BS: {batch_size}"
                )

            # Check for improvement
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                if verbose > 0:
                    _LOG.info(f"  New best model! Val Loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                if verbose > 0:
                    _LOG.info(
                        f"  No improvement for {epochs_without_improvement} epoch(s)"
                    )

            # Early stopping
            if epochs_without_improvement >= patience:
                if verbose > 0:
                    _LOG.info(f"Early stopping after {epoch + 1} epochs")
                break

    except KeyboardInterrupt:
        _LOG.info("Training interrupted by user. Saving best model...")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose > 0:
            _LOG.info(f"Restored best model with Val Loss: {best_val_loss:.4f}")

    model.eval()
    return model, train_loss_history, val_loss_history, vocab_embedders


def _process_batch(
    model: TabCAN,
    batch: dict,
    embedder: TextEmbedder,
    device: torch.device,
    dataset,
    training: bool,
    is_vocab_mode: bool = False,
    vocab_embedders: dict[str, VocabEmbedder] | None = None,
    shuffle_columns: bool = False,
    col_rng: random.Random | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Process a batch of rows from potentially different tables.

    Uses efficient single forward pass with compute_loss_all_columns.
    TextEmbedder handles caching internally.

    Returns:
        Tuple of (total_loss, num_predictions)
    """
    values = batch["values"]  # List[List[str]] padded
    table_ids = batch["table_ids"]  # List[str]

    embed_dim = model.config.embedding_dim

    total_loss = torch.tensor(0.0, device=device, requires_grad=training)
    num_predictions = 0

    # Group rows by table_id for efficient processing
    unique_table_ids = list(set(table_ids))

    for table_id in unique_table_ids:
        # Get table metadata from dataset
        if hasattr(dataset, 'get_table_metadata'):
            meta = dataset.get_table_metadata(table_id)
            columns = meta["columns"]
            class_values = meta["class_values"]
        else:
            columns = dataset.get_table_columns(table_id)
            class_values = dataset.get_class_values(table_id)

        num_cols = len(columns)

        # Get indices of rows belonging to this table
        table_row_indices = [i for i, tid in enumerate(table_ids) if tid == table_id]
        table_batch_size = len(table_row_indices)

        # Get data for this table
        table_values = [values[i][:num_cols] for i in table_row_indices]

        if is_vocab_mode:
            # VocabEmbedders are pre-created before training
            table_embedder = vocab_embedders[table_id]

            # Column embeddings from TextEmbedder (for semantic signal)
            col_embeddings = embedder.embed(columns).to(device)

            # Label embeddings from VocabEmbedder (same space as value embeddings)
            label_embeddings = {}
            value_to_idx = {}
            for col in columns:
                vals = [str(v) for v in class_values[col]]
                label_embeddings[col] = table_embedder.embed_values(col, vals)
                value_to_idx[col] = {v: i for i, v in enumerate(vals)}

            # Optionally shuffle column order for this batch
            if shuffle_columns and col_rng is not None:
                perm = list(range(num_cols))
                col_rng.shuffle(perm)
                batch_columns = [columns[i] for i in perm]
                batch_col_emb = col_embeddings[perm]
                batch_label_emb = {batch_columns[i]: label_embeddings[columns[perm[i]]] for i in range(num_cols)}
                batch_value_to_idx = {batch_columns[i]: value_to_idx[columns[perm[i]]] for i in range(num_cols)}
                # Reorder values according to permutation
                table_values = [[row[i] for i in perm] for row in table_values]
            else:
                batch_columns = columns
                batch_col_emb = col_embeddings
                batch_label_emb = label_embeddings
                batch_value_to_idx = value_to_idx

            # Build value embeddings from VocabEmbedder
            value_emb = torch.zeros(table_batch_size, num_cols, embed_dim, device=device)
            for col_idx in range(num_cols):
                col = batch_columns[col_idx]
                col_values = [table_values[row_idx][col_idx] for row_idx in range(table_batch_size)]
                value_emb[:, col_idx, :] = table_embedder.embed_values(col, col_values)
        else:
            # Text mode: all embeddings from TextEmbedder
            col_embeddings = embedder.embed(columns).to(device)

            label_embeddings = {}
            value_to_idx = {}
            for col in columns:
                vals = [str(v) for v in class_values[col]]
                label_embeddings[col] = embedder.embed(vals).to(device)
                value_to_idx[col] = {v: i for i, v in enumerate(vals)}

            # Optionally shuffle column order for this batch
            if shuffle_columns and col_rng is not None:
                perm = list(range(num_cols))
                col_rng.shuffle(perm)
                batch_columns = [columns[i] for i in perm]
                batch_col_emb = col_embeddings[perm]
                batch_label_emb = {batch_columns[i]: label_embeddings[columns[perm[i]]] for i in range(num_cols)}
                batch_value_to_idx = {batch_columns[i]: value_to_idx[columns[perm[i]]] for i in range(num_cols)}
                # Reorder values according to permutation
                table_values = [[row[i] for i in perm] for row in table_values]
            else:
                batch_columns = columns
                batch_col_emb = col_embeddings
                batch_label_emb = label_embeddings
                batch_value_to_idx = value_to_idx

            # Embed all values (cached by TextEmbedder)
            flat_values = [v for row in table_values for v in row]
            value_emb = embedder.embed(flat_values).to(device)
            value_emb = value_emb.view(table_batch_size, num_cols, embed_dim)

        # Single forward pass for all columns
        loss = model.compute_loss_all_columns(
            value_emb=value_emb,
            col_emb=batch_col_emb,
            label_embeddings=batch_label_emb,
            value_to_idx=batch_value_to_idx,
            columns=batch_columns,
            values=table_values,
            is_text_mode=not is_vocab_mode,
        )

        total_loss = total_loss + loss * table_batch_size * num_cols
        num_predictions += table_batch_size * num_cols

    return total_loss, num_predictions


def save_checkpoint(
    model: TabCAN,
    path: str,
    embedder: TextEmbedder | None = None,
    vocab_embedders: dict[str, VocabEmbedder] | None = None,
    embedding_mode: str = "text",
) -> None:
    """Save model checkpoint."""
    from pathlib import Path as PathLib

    path = PathLib(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "embedding_mode": embedding_mode,
    }
    if embedder is not None:
        checkpoint["text_embedder_state"] = embedder.state_dict()
    if vocab_embedders is not None:
        checkpoint["vocab_embedders_state"] = {
            tid: emb.state_dict() for tid, emb in vocab_embedders.items()
        }

    torch.save(checkpoint, path)
    _LOG.info(f"Saved checkpoint to {path}")
