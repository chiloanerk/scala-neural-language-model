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

- Full training throughput (business corpus, GPU fp64): ~`1100 ex/s`
- Benchmark (`sample=5000`, batch `128`, `fp32`):
  - CPU: `136.0 ex/s`
  - GPU: `1299.6 ex/s`
  - Speedup: `9.55x`

Latest recorded performance (March 22, 2026, from `data/metrics`):

- Train (GPU fp32): `997.72 ex/s` (`train-2026-03-22T03-30-19.508107Z-d1c4cd5f`)
- Previous train baseline: `970.80 ex/s` (`train-2026-03-22T03-29-16.680726Z-0e355956`)
- Train delta vs baseline: `+2.77%`
- Benchmark (CPU fp64, label `smoke`): `1264.17 ex/s` (`benchmark-2026-03-22T03-08-39.129693Z-463af53a`)

This section is a **periodic milestone snapshot** (not updated after every run).

## Requirements

- Java 21+
- SBT 1.10+
- Scala 3.3.3

## Quick Start

### 1) Main menu

```bash
sbt "run"
```

If sbt asks for a main class, choose `app.Main`.
To skip that prompt, run `sbt "run ..."` directly.
The launcher stays open after each action and returns to the menu until you choose `Exit`.
In guided menus (`run`, `train`, `benchmark`, `chunk`), pressing Enter accepts the `[default]` value.

### 2) Train (interactive)

```bash
sbt "run train"
```

### 3) Train (non-interactive)

```bash
sbt "run train --input data/corpus/example-corpus.txt --preset balanced --yes --contextSize 3 --maxVocab 3000"
```

### 3b) Continual replay training (multi-corpus)

```bash
sbt "run train --inputs data/corpus/a.txt,data/corpus/b.txt --inputWeights 0.7,0.3 --replayRatio 0.3 --replayBufferSize 10000 --yes"
```

### 4) Predict

```bash
sbt 'run predict --context "the company" --topK 5'
```

### 5) Benchmark (runs full matrix by default)

```bash
sbt "run benchmark --input data/corpus/example-corpus.txt --sample 2000"
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
sbt "run gpu-info --precision fp64"
```

If GPU/JNI is unavailable, GPU selection safely falls back to CPU with diagnostics.

## Command Reference

### `train`

```bash
sbt "run train [options]"
```

Key options:

- `--input FILE`
- `--inputs CSV` (multi-corpus, comma-separated)
- `--inputWeights CSV` (optional per-input weights, normalized)
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
- `--replayRatio VALUE` (default: `0.3`)
- `--replayBufferSize N` (default: `0`, disables replay persistence)
- `--replayBufferPath FILE` (optional; default `data/models/latest.replay` when replay is enabled)
- `--ewcLambda VALUE` (default: `0.0`, off)
- `--ewcSamples N` (optional EWC sampling count)
- `--yes` (auto-confirm final prompt)

Preset learning rates:

- `quick`: `0.05`
- `balanced`: `0.02`
- `thorough`: `0.01`

Notes:

- Model files are persistent: `data/models/latest.ckpt` and `data/models/latest.vocab`
- `--fresh` is enough to restart from scratch; manual deletion is not required
- With `--fresh --yes`, include `--contextSize` and `--maxVocab` for fully non-interactive runs
- Training output uses a live progress display in interactive terminals (static epoch rows + trajectory status)
- If your terminal does not render live updates, set `TRAIN_PROGRESS_FORCE_TTY=1` (or disable with `TRAIN_PROGRESS_FORCE_TTY=0`)
- If training input is not valid UTF-8, the CLI asks after `Start training?` whether to create a UTF-8 copy (`*.utf8.txt`) and use it for that run
- If you accept, the copy is written once and reused on later runs; phase labels show the effective input file (for example `bbc-all.utf8.txt`)
- If you run with `--yes`, UTF-8 conversion prompt is skipped and training falls back to system-default decoding
- Replay UX in interactive mode:
  - fresh runs ask whether to enable replay memory for future continual training
  - continue runs show replay options (defaults, disable, or customize ratio/buffer)
  - `replayRatio=0.3` is the default/recommended balance
- Interactive `train` supports `b` / `back` on setup prompts to return to the main launcher.

Interactive flow (`sbt "run train"`):

1. Continue existing model or start fresh
2. Select training file
3. Select preset
4. Replay prompt (fresh vs continue flow)
5. Context size / max vocab for fresh architecture
6. Summary + `Start training?` confirmation
7. Optional UTF-8 normalization prompt if decode fails
8. After completion/cancel, launcher remains open for next action

### `predict`

```bash
sbt 'run predict --context "your text" --topK 5'
```

Options:

- `--context TEXT`
- `--topK N`
- `--backend cpu|gpu` (default `gpu`)
- `--precision fp64|fp32` (default `fp64`)

Notes:

- Without `--context`, prediction stays in an interactive loop until `quit`/`exit`.
- High `<UNK>` confidence triggers a hint to use longer in-domain context or larger vocab.

### `chunk`

```bash
sbt "run chunk --input data/corpus/large.txt --lines 2000 --yes"
```

Options:

- `--input FILE`
- `--lines N`
- `--output DIR`
- `--name BASE`
- `--yes`

### `benchmark`

```bash
sbt "run benchmark --input data/corpus/example-corpus.txt --sample 2000"
```

`sbt "run benchmark"` (with no flags) launches an interactive setup that lets you pick backend/precision combinations and optional metrics reporting options before running.
Interactive `benchmark` supports `b` / `back` at menu checkpoints to return to the main launcher.

Options:

- `--input FILE`
- `--sample N`
- `--contextSize N`
- `--maxVocab N`
- `--batchSize N`
- `--backend cpu|gpu` (optional filter; default runs both)
- `--precision fp64|fp32` (optional filter; default runs both)
- `--metrics` (runs complete metrics flow: persist + report output)
- `--metricsDir DIR` (default: `data/metrics`, used with `--metrics`)
- `--runLabel TEXT` (optional label for grouping/comparison)
- `--compareTo latest|RUN_ID|LABEL` (optional baseline selector; default with `--metrics`: `latest`)
- `--regressionWarnPct VALUE` (default: `5.0`, warn-only)

## Metrics Workflow

- Each `train` / `benchmark` run appends one JSON record to `data/metrics/runs.jsonl`.
- A compact index for baseline resolution is maintained in `data/metrics/runs-index.tsv`.
- Human-readable outputs:
  - `data/metrics/latest-summary.txt`
  - `data/metrics/latest-diff-summary.txt` (when a baseline is available)
- Throughput regression policy is warn-only by default: warning at `>5%` slowdown.
- GPU usage truth includes requested backend, effective backend, enabled GPU ops, diagnostics, and backend profile summary.
- Platform metadata is recorded per run (`os`, `os version`, `arch`, `java version`, and detected device name when available).
- Memory captures JVM heap/non-heap plus process RSS at run start/end, with epoch snapshots contributing to peak values during training.
- GC observability includes run-level collection count/time deltas.

### Performance Summary (Example Output)

```text
Run summary
- platform/device: macOS 14.x | arch=aarch64 | java=21 | device=Apple M1
- backend requested/effective: gpu/gpu
- precision: fp32
- examples/sec: 997.72
- total runtime: 41.50s
- top time shares: matMul 58.0%, softmaxBatch 26.5%, ceBatch 3.5%
- fallback ops: none
- memory peak RSS: 2160377856 bytes
- validation: val_loss=6.3442, val_ppl=569.16
- regression vs baseline: +2.77% (status=ok)
```

### Benchmark Update Policy

- Day-to-day metrics stay local in `data/metrics` and are inspected through `benchmark --metrics`.
- README performance numbers are updated only at significant milestones/breakthroughs.
- Typical update triggers:
  - model/training architecture changes
  - backend or hardware-path changes
  - observability/regression framework upgrades
  - material deltas on key metrics (guideline: `>=5-10%`)
- When README numbers are refreshed, include date and run ID(s).

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
sbt "run train --input data/corpus/bbc-business.txt --preset balanced --fresh --yes --contextSize 3 --maxVocab 3000 --precision fp64 --lr 0.02"
```

## How Checkpoints Work

- Training always saves to:
  - `data/models/latest.ckpt`
  - `data/models/latest.vocab`
- Replay memory (when enabled) saves to:
  - `data/models/latest.replay`
- Continuing training reuses existing model/vocab
- `--fresh` starts a new model and overwrites `latest.*` after training
- Interactive interrupt flow allows save choice (best/current/discard) before exit

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
