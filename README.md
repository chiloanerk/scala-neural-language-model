# Scala Neural Language Model (NLM)

A from-scratch neural next-word predictor in Scala 3.

This project trains a small language model on plain text and predicts likely next words for a given context. It includes:

- Pure Scala math/model code (no ML frameworks)
- CLI workflow for train, predict, chunk, gpu-info, and benchmark
- CPU backend + optional Apple Metal GPU backend via JNI
- Batched training path with progress, early stopping, checkpoints

## Status

- Primary tested platform: **Apple MacBook Air (M1, macOS)**
- Other hardware/platforms may work, but are not validated yet

Observed local performance on M1 (example runs):

- Full training throughput (business corpus): ~`2600-3200 ex/s`
- Benchmark (`sample=5000`, batch `128`, `fp32`):
  - CPU: `136.0 ex/s`
  - GPU: `1299.6 ex/s`
  - Speedup: `9.55x`

## Requirements

- Java 21+
- SBT 1.10+
- Scala 3.3.3

## Quick Start

### 1) Main menu

```bash
sbt "runMain app.Main"
```

### 2) Train (interactive)

```bash
sbt "runMain app.Main train"
```

### 3) Train (non-interactive)

```bash
sbt "runMain app.Main train --input data/corpus/example-corpus.txt --preset balanced --yes --contextSize 3 --maxVocab 3000"
```

### 4) Predict

```bash
sbt "runMain app.Main predict --context 'the company' --topK 5"
```

### 5) Benchmark (runs full matrix by default)

```bash
sbt "runMain app.Main benchmark --input data/corpus/example-corpus.txt --sample 2000"
```

By default benchmark runs:

- CPU fp64
- CPU fp32
- GPU fp64
- GPU fp32

### 6) Run tests

```bash
sbt test
```

## GPU (Metal/JNI)

Build optional native bridge:

```bash
metal-jni/scripts/build-metal-jni.sh
```

Probe availability:

```bash
sbt "runMain app.Main gpu-info --precision fp64"
```

If GPU/JNI is unavailable, GPU selection safely falls back to CPU with diagnostics.

## Command Reference

### `train`

```bash
sbt "runMain app.Main train [options]"
```

Key options:

- `--input FILE`
- `--preset quick|balanced|thorough` (default: `balanced`)
- `--fresh` (start new model, ignore existing checkpoint)
- `--contextSize N`
- `--maxVocab N`
- `--backend cpu|gpu` (default: `gpu`)
- `--precision fp64|fp32` (default: `fp64`)
- `--lr VALUE` (override learning rate)
- `--lrDecay VALUE` (default `1.0`)
- `--batchSize N`
- `--prefetch N`
- `--profileGpu`
- `--gpuInfo`
- `--yes` (auto-confirm final prompt)

Preset learning rates:

- `quick`: `0.05`
- `balanced`: `0.02`
- `thorough`: `0.01`

Notes:

- Model files are persistent: `data/models/latest.ckpt` and `data/models/latest.vocab`
- `--fresh` is enough to restart from scratch; manual deletion is not required
- With `--fresh --yes`, include `--contextSize` and `--maxVocab` for fully non-interactive runs

### `predict`

```bash
sbt "runMain app.Main predict --context 'your text' --topK 5"
```

Options:

- `--context TEXT`
- `--topK N`
- `--backend cpu|gpu` (default `gpu`)
- `--precision fp64|fp32` (default `fp64`)

### `chunk`

```bash
sbt "runMain app.Main chunk --input data/corpus/large.txt --lines 2000 --yes"
```

Options:

- `--input FILE`
- `--lines N`
- `--output DIR`
- `--name BASE`
- `--yes`

### `benchmark`

```bash
sbt "runMain app.Main benchmark --input data/corpus/example-corpus.txt --sample 2000"
```

Options:

- `--input FILE`
- `--sample N`
- `--contextSize N`
- `--maxVocab N`
- `--batchSize N`
- `--backend cpu|gpu` (optional filter; default runs both)
- `--precision fp64|fp32` (optional filter; default runs both)

## Training Data Guide

### Example workflow: BBC data from Kaggle website

1. Download a BBC text dataset zip from [Kaggle](https://www.kaggle.com/) in your browser.
2. Save it under `data/corpus/`.
3. Unzip:

```bash
unzip data/corpus/<downloaded-file>.zip -d data/corpus
```

4. Combine the business folder into one training file:

```bash
cat data/corpus/bbc/business/*.txt > data/corpus/bbc-business.txt
```

5. Start fresh training:

```bash
sbt "runMain app.Main train --input data/corpus/bbc-business.txt --preset balanced --fresh --yes --contextSize 3 --maxVocab 3000 --precision fp64 --lr 0.02"
```

## How Checkpoints Work

- Training always saves to:
  - `data/models/latest.ckpt`
  - `data/models/latest.vocab`
- Continuing training reuses existing model/vocab
- `--fresh` starts a new model and overwrites `latest.*` after training

## Data/Repo Policy

Current repo policy keeps only:

- `data/corpus/example-corpus.txt`

Large corpora, chunks, and model artifacts are git-ignored.

## Project Layout

```text
data/
  corpus/
    example-corpus.txt
  models/
    latest.ckpt
    latest.vocab
src/main/scala/
  app/        # CLI
  data/       # tokenization, vocab IO
  linalg/     # matrix/vector ops
  nn/         # model forward/backward/update
  compute/    # cpu/gpu backend abstraction
  train/      # trainer + checkpoint IO
src/test/scala/
  ...         # unit and regression tests
metal-jni/
  ...         # Metal bridge (optional)
```

## Notes

- This is an educational/research-style project, not a production LLM stack.
- Quality depends heavily on corpus quality and training setup.
- Use validation loss/perplexity for quality decisions; examples/sec is speed only.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the full license text in [`LICENSE`](LICENSE).
