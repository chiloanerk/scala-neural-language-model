# Scala 3 Next-Word Predictor - Project Context

## Quick Start

```bash
# 1. Interactive training (recommended)
sbt "runMain app.Main train"

# 2. Quick command-line training
sbt "runMain app.Main train --preset balanced --input data/2026-03-17/example-corpus.txt"

# 3. Predict next words
sbt "runMain app.Main predict --context 'the cat' --topK 5"
```

## Project Overview

This is a **pure Scala 3 implementation of a neural language model** built from scratch without any ML libraries.

**What it does:** Given some text context, predicts the most likely next word.

**Architecture:** Embedding layer → Hidden layer (Tanh) → Output layer (Softmax)

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
│   │   ├── Trainer.scala        # Training loop with early stopping, progress bar
│   │   └── CheckpointIO.scala   # Model checkpoint save/load
│   ├── eval/
│   │   └── Metrics.scala        # Loss and perplexity calculation
│   └── app/
│       ├── Main.scala           # CLI entry point (interactive train/predict)
│       └── TestRunner.scala     # Built-in test suite
```

## Usage

### Training

**Interactive (Recommended):**
```bash
sbt "runMain app.Main train"
```

Guides you through:
1. Selecting training text file
2. Choosing quality preset (Quick/Balanced/Thorough)
3. Optional advanced options
4. Confirming and starting training

**Command-line:**
```bash
# Quick test
sbt "runMain app.Main train --preset quick --input corpus.txt"

# Standard
sbt "runMain app.Main train --preset balanced --input corpus.txt"

# Best quality
sbt "runMain app.Main train --preset thorough --input corpus.txt"
```

### Prediction

**Interactive:**
```bash
sbt "runMain app.Main predict"
```

**Command-line:**
```bash
sbt "runMain app.Main predict --context 'the cat sat' --topK 5"
```

### Testing
```bash
sbt "runMain app.TestRunner"
```

## Training Presets

| Preset | Epochs | Patience | Hidden | Embed | Time | Use Case |
|--------|--------|----------|--------|-------|------|----------|
| quick | 5 | 3 | 32 | 16 | ~30s | Testing |
| balanced | 20 | 5 | 64 | 24 | ~2min | Default |
| thorough | 50 | 10 | 128 | 48 | ~10min | Best quality |

## Key Concepts (Simple Explanations)

| Term | What It Means |
|------|---------------|
| **Model** | The "trained brain" - learned word patterns |
| **Vocabulary** | List of words the model knows |
| **Epoch** | One full pass through training data |
| **Patience** | Stop if no improvement after N epochs (prevents overfitting) |
| **Context Size** | How many previous words to look at |
| **Embedding Dim** | How "rich" word representations are |
| **Hidden Dim** | Model's "thinking power" |

## Key Implementation Details

### Linear Algebra (`linalg/`)
- `Matrix`: Flat `Vector[Double]` storage with row/col metadata and transpose view
- `Vec`: Type alias for `Vector[Double]`
- Stable softmax using max-shift for numerical stability
- Operations: `vecAdd`, `vecSub`, `scalarMul`, `dot`, `matVecMul`, `outer`, `tanhVec`, `relu`, `reluGrad`

### Neural Network (`nn/`)
- **Parameters**: `Params(E, W1, b1, W2, b2)` - Embedding, two weight matrices, two biases
- **Forward**: Returns `ForwardCache` with intermediate values for backprop
- **Backward**: Manual gradient computation with embedding gradient scatter accumulation
- **Activation**: Configurable (tanh or relu)

### Data Pipeline (`data/`)
- Tokenization: Lowercase, split on non-alphanumeric
- Vocabulary: Frequency-based selection with `<UNK>` token
- Examples: Sliding window of context→target pairs
- Split: Deterministic shuffle with seed

### Training (`train/`)
- SGD with optional gradient clipping and L2 regularization
- **Early stopping**: Stops when validation loss stops improving
- **Progress bar**: Shows ETA, throughput, and current loss

### Testing
- Custom `TestRunner` with 13 tests (no external test framework)
- Tests cover: matrix ops, softmax, forward shapes, gradient accumulation, gradient checking, ReLU, early stopping, training regression, inference

## Development Conventions

- **No external ML libraries**: All neural network operations implemented from scratch
- **Functional style**: Immutable case classes, pure functions where possible
- **Explicit types**: Full type annotations on public APIs
- **Deterministic training**: Seed-based reproducibility for all random operations
- **Interactive-first**: Default to user-friendly interactive mode

## File Formats

### Model Checkpoint (`*.ckpt`)
Text format containing:
- Model configuration (contextSize, embedDim, hiddenDim, vocabSize)
- All parameter matrices (E, W1, b1, W2, b2)

### Vocabulary (`*.txt`)
Plain text, one word per line:
```
<UNK>
the
and
cat
dog
...
```

## Git Ignore Rules

```
# Keep example corpus
!data/**/example-corpus.txt

# Ignore generated files
data/**/*.ckpt
data/**/*.txt (except example-corpus.txt)
```

## Troubleshooting

### Context not recognized (all `<UNK>` tokens)
If you see `context_ids=[0,0,0]` in prediction output, the words aren't in the vocabulary. Use words from your training data.

### Model overfitting
If validation loss increases while training loss decreases:
- The **early stopping** (patience) should catch this automatically
- Or use a smaller preset (less epochs)
- Or add more training data

### Slow training
- Use `--preset quick` for testing
- Reduce model size (advanced: `--hiddenDim 32 --embedDim 16`)
- This is CPU-only (no GPU acceleration)
