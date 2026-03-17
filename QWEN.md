# Scala 3 Next-Word Predictor - Project Context

## Quick Start

```bash
# 1. Run tests
sbt "runMain app.TestRunner"

# 2. Train a model (using included example corpus)
sbt "runMain app.Main train --input data/2026-03-17/example-corpus.txt --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --epochs 10"

# 3. Predict next words
sbt 'runMain app.Main predict --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --context "the cat" --topK 5'
```

## Project Overview

This is a **pure Scala 3 implementation of a neural language model** built from scratch without any ML libraries. The project implements a next-word prediction model using:

- **Architecture**: Embedding layer → Hidden layer (Tanh activation) → Output layer (Softmax)
- **Manual backpropagation**: Custom forward/backward pass implementation
- **Flat matrix representation**: `Matrix` type using `Vector[Double]` with transpose view (no eager copy)
- **Training**: SGD with optional gradient clipping, L2 regularization, and learning rate decay

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Scala 3.3.3 |
| Build Tool | SBT 1.10.2 |
| Dependencies | None (pure Scala standard library) |

## Project Structure

```
.
├── data/                       # Training data and model outputs
│   └── YYYY-MM-DD/            # Date-organized folders
│       ├── corpus.txt         # Input training text
│       ├── model.ckpt         # Trained model checkpoint
│       └── vocab.txt          # Vocabulary file
├── src/main/scala/
│   ├── linalg/
│   │   ├── Types.scala          # Matrix and Vec type definitions
│   │   └── LinearAlgebra.scala  # Matrix/vector operations, stable softmax
│   ├── data/
│   │   ├── TextPipeline.scala   # Tokenization, vocab building, dataset creation
│   │   └── VocabIO.scala        # Vocabulary save/load
│   ├── nn/
│   │   └── LanguageModel.scala  # Model definition, forward/backward, parameter updates
│   ├── train/
│   │   ├── Trainer.scala        # Training loop with epoch metrics
│   │   └── CheckpointIO.scala   # Model checkpoint save/load
│   ├── eval/
│   │   └── Metrics.scala        # Loss and perplexity calculation
│   └── app/
│       ├── Main.scala           # CLI entry point (train/predict/test)
│       └── TestRunner.scala     # Built-in test suite (no test framework)
```

## Building and Running

### Compile
```bash
sbt compile
```

### Run All Tests
```bash
sbt "runMain app.TestRunner"
```

### Train a Model
```bash
sbt "runMain app.Main train --input data/2026-03-17/example-corpus.txt --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt \
  --contextSize 3 --embedDim 24 --hiddenDim 64 --maxVocab 3000 \
  --epochs 10 --lr 0.05 --lrDecay 1.0 --l2 0.0 --clipNorm 1.0 \
  --seed 42 --trainRatio 0.9"
```

### Predict Next Word

**Important:** Use single quotes around the full command and double quotes for the context string:

```bash
# Single word context
sbt "runMain app.Main predict --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --context the --topK 5"

# Multi-word context (note the quote style)
sbt 'runMain app.Main predict --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --context "the cat" --topK 5'
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Input text file for training |
| `--model` | `model.ckpt` | Model checkpoint path |
| `--vocab` | `vocab.txt` | Vocabulary file path |
| `--contextSize` | `3` | Context window size |
| `--embedDim` | `24` | Embedding dimension |
| `--hiddenDim` | `64` | Hidden layer dimension |
| `--maxVocab` | `3000` | Maximum vocabulary size |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `0.05` | Learning rate |
| `--lrDecay` | `1.0` | Learning rate decay per epoch |
| `--l2` | `0.0` | L2 regularization coefficient |
| `--clipNorm` | (optional) | Gradient clipping threshold |
| `--seed` | `42` | Random seed |
| `--trainRatio` | `0.9` | Train/validation split ratio |
| `--context` | (required) | Context text for prediction |
| `--topK` | `5` | Number of top predictions to show |

## Key Implementation Details

### Linear Algebra (`linalg/`)
- `Matrix`: Flat `Vector[Double]` storage with row/col metadata and transpose view
- `Vec`: Type alias for `Vector[Double]`
- Stable softmax using max-shift for numerical stability
- Operations: `vecAdd`, `vecSub`, `scalarMul`, `dot`, `matVecMul`, `outer`, `tanhVec`

### Neural Network (`nn/`)
- **Parameters**: `Params(E, W1, b1, W2, b2)` - Embedding, two weight matrices, two biases
- **Forward**: Returns `ForwardCache` with intermediate values for backprop
- **Backward**: Manual gradient computation with embedding gradient scatter accumulation
- **Update**: SGD with optional L2 regularization and gradient clipping

### Data Pipeline (`data/`)
- Tokenization: Lowercase, split on non-alphanumeric
- Vocabulary: Frequency-based selection with `<UNK>` token
- Examples: Sliding window of context→target pairs
- Split: Deterministic shuffle with seed

### Testing
- Custom `TestRunner` with 7 milestone tests (no external test framework)
- Tests cover: matrix ops, softmax stability, forward shapes, gradient accumulation, gradient checking, training regression, inference

## Development Conventions

- **No external ML libraries**: All neural network operations implemented from scratch
- **Functional style**: Immutable case classes, pure functions where possible
- **Explicit types**: Full type annotations on public APIs
- **Deterministic training**: Seed-based reproducibility for all random operations
- **No test framework dependency**: Self-contained test runner for portability

## File Formats

### Model Checkpoint (`*.ckpt`)
Binary format containing model parameters and configuration.

### Vocabulary (`*.txt`)
Text file with token-to-id mappings.

## Troubleshooting

### Context not recognized (all `<UNK>` tokens)
If you see `context_ids=[0,0,0]` in prediction output, the quote style is wrong. Use:
- Single quotes around the full command: `'... --context "text" ...'`
- Double quotes around the context value: `--context "the cat"`

### Model overfitting
If validation loss increases while training loss decreases:
- Add more training data to your corpus
- Reduce model complexity (`--embedDim`, `--hiddenDim`)
- Add regularization (`--l2 0.01 --clipNorm 1.0`)
