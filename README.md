# TabCAN

**Tabular Causal Attention Network**

A simple autoregressive model for tabular data synthesis using causal self-attention and LLM embeddings.

## Features

- **Synthetic data generation** - Generate realistic categorical tabular data
- **Conditional generation** - Seed with fixed column values
- **Multi-table support** - Train on multiple tables, sample from any
- **Flexible embeddings** - Choose between LLM embeddings (semantic) or learnable vocabulary embeddings (faster)
- **Multi-table pretraining** - Pretrain on diverse table datasets (e.g., T4)
- **Lightweight** - ~0.8M parameters (medium)

## Installation

```bash
pip install tabcan
```

Or for development:

```bash
git clone https://github.com/mostly-ai/tabcan.git
cd tabcan
pip install -e ".[dev]"
```

## Quick Start

```python
from tabcan import TabCAN
import pandas as pd

# Load your data
df = pd.DataFrame({
    "color": ["red", "blue", "green", "red", "blue"],
    "size": ["S", "M", "L", "M", "S"],
    "price": ["low", "high", "medium", "low", "high"],
})

# Fit model (pass tables as a dictionary)
model = TabCAN(max_epochs=100, verbose=1)
model.fit({"products": df})

# Generate synthetic data
synthetic = model.sample(n_samples=100, table_id="products")
print(synthetic)
```

## Conditional Generation

You can seed generation with fixed column values:

```python
# Generate samples with fixed "color"
seed_data = pd.DataFrame({"color": ["red", "blue", "green"]})
synthetic = model.sample(n_samples=3, table_id="products", seed_data=seed_data)
```

## Model Sizes

TabCAN model sizes differ in hidden size and attention heads:

| Size   | Hidden | Heads | Parameters |
|--------|--------|-------|------------|
| small  | 128    | 4     | ~0.3M      |
| medium | 256    | 8     | ~0.8M      |
| large  | 512    | 8     | ~2.0M      |

```python
model = TabCAN(model_size="large")
```

## Embedding Modes

TabCAN supports two embedding modes for representing values and column names:

| Mode | Description | Training | Generalization |
|------|-------------|----------|----------------|
| `text` | Pre-trained LLM embeddings (MiniLM, 384-dim) | Frozen | Can generalize to unseen values |
| `vocab` | Per-column learnable vocabulary embeddings | Trained with model | No generalization to OOV |

### Text Mode (default)

Uses sentence-transformers to embed values and column names semantically. Good for:
- Small datasets where semantic similarity helps
- When you want to leverage pre-trained knowledge
- Zero-shot or few-shot scenarios

```python
model = TabCAN(embedding_mode="text")  # default
```

### Vocab Mode

Uses learnable `nn.Embedding` lookup tables for **values** with text-based **column embeddings**:
- Value embeddings: per-column learnable lookup tables (384-dim)
- Column embeddings: LLM-based (same as text mode, for target signal)

Good for:
- Large datasets with many unique values
- Faster training (no LLM inference for values)
- Retains semantic understanding of column names

```python
model = TabCAN(embedding_mode="vocab")
```

**Note:** Vocab mode builds a separate vocabulary for each column during `fit()`. Each table has its own set of per-column value embeddings. Unknown values at sampling time are mapped to an `<UNK>` token.

## Save/Load

```python
# Save
model.save("tabcan_model.pt")

# Load
model = TabCAN.load("tabcan_model.pt")
```

## Architecture

### Overview

- **Autoregressive**: Columns generated left-to-right, each conditioned on previous columns
- **Causal attention**: Enables efficient parallel training (single forward pass for all columns)
- **Cosine similarity**: Predictions via similarity to candidate value embeddings
- **Column shuffling**: Optional order invariance via `shuffle_columns=True`

### Model Components

During training, all columns are processed in parallel with a causal mask:

```
Input sequence (shifted) - both modes:
[START, val0+col0, val1+col1, ..., val_{n-2}+col_{n-2}]

  Text mode:  val_i = LLM(value_text), col_i = LLM(column_name)
  Vocab mode: val_i = ValueEmb[col][val_idx], col_i = LLM(column_name)

   |           |            |                  |
   v           v            v                  v
+--------------------------------------------------+
|  Input Projections: value_proj + column_proj     |
|  (384-dim -> hidden_size, e.g. 256)              |
+--------------------------------------------------+
   |           |            |                  |
   v           v            v                  v
+--------------------------------------------------+
|              Dropout + LayerNorm                 |
+--------------------------------------------------+
   |           |            |                  |
   v           v            v                  v
+--------------------------------------------------+
|            Causal Self-Attention                 |
|  (position i attends to positions 0..i-1 only)   |
+--------------------------------------------------+
   |           |            |                  |
   v           v            v                  v
+--------------------------------------------------+
|         Residual + LayerNorm                     |
+--------------------------------------------------+
   |           |            |                  |
   v           v            v                  v
  [h0,        h1,          h2,       ...,    h_{n-1}]
   |           |            |                  |
   + concat    + concat     + concat           + concat
   |           |            |                  |
 [tgt0,      tgt1,        tgt2,      ...,   tgt_{n-1}]  <- target_proj(col_emb)
   |           |            |                  |
   v           v            v                  v
+--------------------------------------------------+
|     Output Projection (hidden*2 -> embed_dim)    |
+--------------------------------------------------+
   |           |            |                  |
   v           v            v                  v
+--------------------------------------------------+
|       Cosine Similarity with candidate values    |
+--------------------------------------------------+
   |           |            |                  |
   v           v            v                  v
 [pred0,     pred1,       pred2,     ...,  pred_{n-1}]
   |           |            |                  |
Targets:     val0,        val1,      ...,    val_{n-1}

Target column signal (both modes):
  tgt_i = target_proj(col_i)  (projected LLM column embedding)
```

Each position predicts its column value using attended context from previous positions plus a target column signal.

## Training

- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
- **Early Stopping**: Stops when validation loss doesn't improve for `patience` epochs
- **Best Checkpoint**: Returns model with best validation loss
- **Weight Decay**: L2 regularization, scaled inversely with dataset size
- **Column Shuffling**: Shuffle column order each batch (enabled by default)
- **Batch Size**: Auto-scaled based on dataset size

```python
model = TabCAN(
    model_size="medium",
    max_epochs=100,
    learning_rate=1e-3,
    patience=20,
    verbose=1,
)
model.fit({"my_table": df})
```

## Sampling

### Temperature Control

Higher temperature = more random sampling:

```python
# More deterministic
synthetic = model.sample(n_samples=100, table_id="my_table", temperature=0.5)

# More random
synthetic = model.sample(n_samples=100, table_id="my_table", temperature=1.5)
```

## Pretraining on T4

Pretrain TabCAN on the T4 dataset (3.1M tables from TabLib) to improve performance on small datasets.

### Step 1: Download tables

```bash
# Download 1,000 tables to ~/.cache/tabcan/t4
uv run python scripts/download_t4.py --num_tables 1000
```

### Step 2: Pretrain

```bash
uv run python scripts/pretrain_t4.py \
    --checkpoint checkpoints/t4_1k.pt \
    --num_tables 1000 \
    --model_size medium \
    --batch_size 256 \
    --max_epochs 3
```

### Step 3: Fine-tune on your data

```python
from tabcan import TabCAN

# Load pretrained and fine-tune
model = TabCAN(checkpoint="checkpoints/t4_1k.pt", learning_rate=1e-4)
model.fit({"my_table": your_df})

# Generate synthetic data
synthetic = model.sample(n_samples=1000, table_id="my_table")
```

### Pretraining Benefits

- Better performance on small datasets (100-1000 rows)
- Faster convergence during fine-tuning
- Captures general tabular patterns from diverse tables

## API Reference

### TabCAN

```python
class TabCAN:
    def __init__(
        self,
        model_size="medium",       # "small", "medium", "large"
        checkpoint=None,           # Path to pretrained model for fine-tuning
        embedding_mode="text",     # "text" (LLM) or "vocab" (learnable)
        max_epochs=100,
        batch_size=512,            # Auto-scaled based on dataset size
        learning_rate=1e-3,
        patience=20,
        lr_patience=2,
        lr_factor=0.9,
        weight_decay=0.001,        # Auto-scaled based on dataset size
        val_split=0.2,
        shuffle_columns=True,
        random_state=None,
        verbose=0,
    ): ...

    def fit(
        self,
        tables: dict[str, pd.DataFrame],  # {"table_id": df, ...}
        y=None,
        reset=True,                # True=fresh start, False=add tables to existing model
    ) -> self

    def sample(
        self,
        n_samples=1,
        table_id=None,             # Required if multiple tables registered
        seed_data=None,
        temperature=1.0,
    ) -> pd.DataFrame

    def save(self, path) -> None

    @classmethod
    def load(cls, path) -> TabCAN
```

### Multi-Table Support

Train on multiple tables and sample from any:

```python
model = TabCAN()
model.fit({
    "customers": df_customers,
    "products": df_products,
})

# Sample from specific table
syn_customers = model.sample(n_samples=100, table_id="customers")
syn_products = model.sample(n_samples=100, table_id="products")

# Add more tables without resetting
model.fit({"orders": df_orders}, reset=False)
```

**Note:** When only one table is registered, `table_id` can be omitted from `sample()`.

## License

MIT
