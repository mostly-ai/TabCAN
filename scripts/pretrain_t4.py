#!/usr/bin/env python
"""
Pretrain TabCAN on T4 dataset.

Usage:
    # Pretrain on 1000 tables
    python scripts/pretrain_t4.py --checkpoint checkpoints/t4_1k.pt --num_tables 1000

    # Pretrain on 10000 tables with larger model
    python scripts/pretrain_t4.py \
        --checkpoint checkpoints/t4_10k.pt \
        --num_tables 10000 \
        --model_size large \
        --batch_size 128

    # Resume from checkpoint
    python scripts/pretrain_t4.py \
        --checkpoint checkpoints/t4_final.pt \
        --resume checkpoints/t4_10k.pt \
        --max_epochs 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tabcan.interface import TabCAN
from tabcan.t4_dataset import T4Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pretrain TabCAN on T4 dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to save the pretrained model checkpoint",
    )
    parser.add_argument(
        "--num_tables",
        type=int,
        default=1000,
        help="Number of tables to use for pretraining (default: 1000)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size (default: medium)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum training epochs (default: 3)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="T4 cache directory (default: ~/.cache/tabcan/t4)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--embedding_mode",
        type=str,
        default="text",
        choices=["text", "vocab"],
        help="Embedding mode: 'text' (LLM) or 'vocab' (learnable)",
    )
    args = parser.parse_args()

    # Load dataset
    _LOG.info(f"Loading T4 dataset (max {args.num_tables} tables)...")
    try:
        dataset = T4Dataset(
            cache_dir=args.cache_dir,
            max_tables=args.num_tables,
        )
    except FileNotFoundError as e:
        _LOG.error(str(e))
        _LOG.error("Please run: python scripts/download_t4.py --num_tables <N>")
        sys.exit(1)

    stats = dataset.get_stats()
    _LOG.info(
        f"Dataset: {stats['num_tables']} tables, {stats['total_rows']:,} rows, "
        f"{stats['avg_columns']:.1f} avg columns"
    )

    if stats["total_rows"] == 0:
        _LOG.error("No data found in T4 dataset!")
        _LOG.error("Please download the dataset first:")
        _LOG.error(f"  uv run python scripts/download_t4.py --num_tables {args.num_tables}")
        sys.exit(1)

    # Create TabCAN model
    _LOG.info(f"Creating {args.model_size} TabCAN model...")
    model = TabCAN(
        model_size=args.model_size,
        checkpoint=args.resume,
        embedding_mode=args.embedding_mode,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        val_split=args.val_split,
        verbose=1,
    )

    # Train on T4 dataset
    _LOG.info("Starting training...")
    model.fit(dataset)

    # Save checkpoint
    _LOG.info(f"Saving checkpoint to {args.checkpoint}")
    model.save(args.checkpoint)

    _LOG.info("Training complete!")
    if model.train_loss_history_:
        _LOG.info(f"Final train loss: {model.train_loss_history_[-1]:.4f}")
    if model.val_loss_history_:
        _LOG.info(f"Final val loss: {model.val_loss_history_[-1]:.4f}")


if __name__ == "__main__":
    main()
