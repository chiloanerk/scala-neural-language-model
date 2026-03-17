# Scala 3 Next-Word Predictor (From Scratch)

Pure Scala 3 implementation of a tiny neural language model (no ML libraries), with:
- Flat `Matrix` representation (`Vector[Double] + rows/cols`)
- Transpose view (no eager copy)
- Stable softmax (max-shift)
- Manual forward/backward pass
- Xavier initialization
- Embedding gradient scatter accumulation
- SGD training + optional gradient clipping/L2/lr decay
- Checkpoint + vocab save/load
- CLI for train/predict
- Built-in test runner (no test framework dependency)

## Quick Start

```bash
# 1. Run tests
sbt "runMain app.TestRunner"

# 2. Train a model (using included example corpus)
sbt "runMain app.Main train --input data/2026-03-17/example-corpus.txt --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --epochs 10"

# 3. Predict next words
sbt 'runMain app.Main predict --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --context "the cat" --topK 5'
```

## Commands

Compile:

```bash
sbt compile
```

Run all tests:

```bash
sbt "runMain app.TestRunner"
```

Train:

```bash
sbt "runMain app.Main train --input data/2026-03-17/example-corpus.txt --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt \
  --contextSize 3 --embedDim 24 --hiddenDim 64 --maxVocab 3000 \
  --epochs 10 --lr 0.05 --lrDecay 1.0 --l2 0.0 --clipNorm 1.0 \
  --seed 42 --trainRatio 0.9"
```

Predict:

```bash
# Single word context
sbt "runMain app.Main predict --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --context the --topK 5"

# Multi-word context (use single quotes around command, double quotes for context)
sbt 'runMain app.Main predict --model data/2026-03-17/model.ckpt --vocab data/2026-03-17/vocab.txt --context "the cat" --topK 5'
```

## Project Structure

```
.
├── data/                       # Training data and model outputs
│   └── YYYY-MM-DD/            # Date-organized folders
│       ├── corpus.txt         # Input training text
│       ├── model.ckpt         # Trained model checkpoint
│       └── vocab.txt          # Vocabulary file
└── src/main/scala/
    ├── linalg/                # Matrix operations, stable softmax
    ├── data/                  # Tokenization, vocab, dataset building
    ├── nn/                    # Neural network model
    ├── train/                 # Training loop, checkpoint I/O
    ├── eval/                  # Loss and perplexity metrics
    └── app/                   # CLI and test runner
```

## Main Packages

- `linalg`: flat matrix + vector/matrix ops + stable softmax
- `data`: tokenization, vocab, dataset building, deterministic split, vocab I/O
- `nn`: model contracts, forward cache, backprop, Xavier init, updates
- `train`: trainer loop, checkpoint save/load
- `eval`: loss/perplexity
- `app`: CLI + custom test runner
