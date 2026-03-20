# Scala 3 Next-Word Predictor (From Scratch)

A pure Scala 3 implementation of a neural language model. Train incrementally on your text, predict next words.

## Quick Start

### 1. Show Main Menu

```bash
sbt "runMain app.Main"
```

### 2. Train (or Continue Training)

```bash
sbt "runMain app.Main train"
```

### 3. Predict

```bash
sbt "runMain app.Main predict --context 'the cat sat'"
```

### 4. Split Large Files

```bash
sbt "runMain app.Main chunk --input large-file.txt --lines 1000 --yes"
```

## How It Works

**One persistent model** at `data/models/latest.ckpt` that gets smarter over time:

```bash
# Session 1: Train on 1000 lines
sbt "runMain app.Main train --input part1.txt --preset quick"

# Session 2: Continue training (automatically continues from session 1!)
sbt "runMain app.Main train --input part2.txt --preset quick"

# Your model accumulates knowledge from all sessions!
```

## Commands

### Main Menu

```bash
sbt "runMain app.Main"
```

Shows model status and available commands.

### Train (Interactive)

```bash
# Fully interactive - asks for file, preset, and confirms
sbt "runMain app.Main train"

# Semi-interactive - specify file, asks for rest
sbt "runMain app.Main train --input data/corpus/text.txt"

# Fully automated
sbt "runMain app.Main train --input data.txt --preset balanced --yes"
```

| Option | Description |
|--------|-------------|
| `--input FILE` | Training text (or select interactively) |
| `--preset NAME` | quick/balanced/thorough (default: balanced) |
| `--fresh` | Start over, ignore existing model |
| `--contextSize N` | Words to look back (auto-uses existing model's setting) |
| `--maxVocab N` | Max unique words (auto-uses existing model's setting) |
| `--yes` | Auto-confirm without prompting |

**Note:** When continuing training, the system automatically uses your existing model's architecture (context size, embedding dimensions, etc.). You only need to specify these when using `--fresh`.

### Predict

```bash
# With context
sbt "runMain app.Main predict --context 'hello world' --topK 5"
```

| Option | Description |
|--------|-------------|
| `--context TEXT` | Words to continue from |
| `--topK N` | Predictions to show (default: 5) |

### Chunk (Split Large Files)

```bash
# Interactive - select file and options
sbt "runMain app.Main chunk"

# Command-line with auto-confirm
sbt "runMain app.Main chunk --input large-file.txt --lines 1000 --yes"
```

| Option | Description |
|--------|-------------|
| `--input FILE` | File to split (or select interactively) |
| `--lines N` | Lines per chunk (default: auto-recommend) |
| `--output DIR` | Output directory (default: `<input>/chunks/`) |
| `--name BASE` | Base name for chunks (default: input filename) |
| `--yes` | Auto-confirm without prompting |

### Presets

| Preset | Epochs | Approx. Time (1000 lines) | Use Case |
|--------|--------|---------------------------|----------|
| quick | 5 | ~30 seconds | Rapid testing and iteration |
| balanced | 20 | ~2 minutes | Standard training (default) |
| thorough | 50 | ~10 minutes | High-quality final model |

## Best Practices Guide

### Training Data Organization

```
data/
  corpus/
    01-bbc-news/         ← Organize by source
      business/
      entertainment/
      politics/
    02-books/
      book1.txt
      book2.txt
    example-corpus.txt   ← Your base training data
```

### Recommended Workflow

#### Step 1: Prepare Your Data

```bash
# If you have a large file (>2000 lines), split it:
sbt "runMain app.Main chunk --input data/corpus/train.txt --lines 2000 --yes"

# This creates:
#   data/corpus/chunks/train-part1.txt (2000 lines)
#   data/corpus/chunks/train-part2.txt (2000 lines)
#   ...
```

#### Step 2: Train Incrementally

```bash
# Session 1
sbt "runMain app.Main train --input data/corpus/chunks/train-part1.txt --preset quick"

# Session 2 (automatically continues from session 1!)
sbt "runMain app.Main train --input data/corpus/chunks/train-part2.txt --preset quick"

# Session 3
sbt "runMain app.Main train --input data/corpus/chunks/train-part3.txt --preset quick"
```

#### Step 3: Test Progress

```bash
# After each session, test predictions
sbt "runMain app.Main predict --context 'the cat'"
```

### Data Quality Guidelines

| Quality | Characteristics | Use |
|---------|-----------------|-----|
| **High** | Proper grammar, complete sentences, consistent style | Train first |
| **Medium** | Some errors, informal but readable | Train second |
| **Low** | Many errors, fragments, inconsistent | Train last or skip |

**Good sources:**
- Books (public domain)
- News articles (BBC, Reuters, etc.)
- Well-written blog posts
- Documentation

**Avoid:**
- Social media posts (too noisy)
- Chat logs (incomplete sentences)
- Code mixed with text
- Multiple languages mixed

### Chunk Size Recommendations

| File Size | Chunk Size | Sessions |
|-----------|------------|----------|
| < 1000 lines | Train as-is | 1 session |
| 1000-5000 lines | 1000 lines/chunk | 1-5 sessions |
| 5000-20000 lines | 2000 lines/chunk | 3-10 sessions |
| > 20000 lines | 2000-5000 lines/chunk | Many sessions |

### Training Tips

1. **Start small**: Train on 500-1000 lines first to verify everything works
2. **Quality over quantity**: 1000 lines of good text > 10000 lines of noise
3. **Consistent style**: Keep training data similar in style/topic
4. **One language**: Don't mix languages in the same model
5. **Backup before risky training**:
   ```bash
   cp data/models/latest.ckpt data/models/backup-$(date +%Y%m%d).ckpt
   ```
6. **Use progress bar**: Training shows progress every 10% with ETA

### Understanding Context Size

**Context size = How many previous words the model looks at to predict the next word.**

```
Context size 3:
"the cat sat on the mat"
         ^^^^^
         these 3 words predict → "on"

Context size 5:
"the cat sat on the mat"
         ^^^^^^^^^^^^^^^
         these 5 words predict → "the"
```

| Context Size | Pros | Cons | Best For |
|--------------|------|------|----------|
| Small (2-3) | Faster training, needs less data | Short-term memory only | Simple patterns, chatbots |
| Medium (4-6) | Good balance | Moderate training time | General purpose (recommended) |
| Large (7-10+) | Long-term memory, better coherence | Much slower, needs lots of data | Stories, long documents |

**Note:** Context size is set when you first train and cannot be changed without `--fresh`.

### When to Use `--fresh`

```bash
# Normal: Continue existing model (99% of cases)
sbt "runMain app.Main train --input new-data.txt"

# Fresh start (rare):
# - Switching to completely different domain/topic
# - Starting with much better quality data
# - Model seems "broken" from bad training
# - Changing context size or model architecture
sbt "runMain app.Main train --fresh --input new-domain.txt"
```

### Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| All `<UNK>` predictions | Word not in vocabulary | Use words from training data |
| Nonsense predictions | Model undertrained | Train more sessions |
| Predictions too repetitive | Model overfitting | Use `--fresh` with more diverse data |
| Training very slow | Chunk too large | Use smaller chunks (500-1000 lines) |
| Architecture mismatch error | Changed context size | Use `--fresh` when changing architecture |
| No progress bar shown | File too small (<100 examples) | Progress shows every 10% or 100 examples |

## Project Structure

```
data/
  models/
    latest.ckpt       ← Your model (gets smarter over time)
    latest.vocab      ← Vocabulary
    backups/          ← Optional manual backups
  corpus/
    chunks/           ← Auto-generated chunks (git-ignored)
      train-part1.txt
      train-part2.txt
    bbc/              ← Your raw data
    books/            ← More training data
```

## Git Setup

```gitignore
# In your .gitignore:
data/models/*.ckpt
data/models/*.vocab
data/corpus/chunks/
```

Commit your training data, not the models (they can be regenerated).

## Examples

### Complete Workflow

```bash
# 1. Show menu
sbt "runMain app.Main"

# 2. You have a large file (36000 lines) - split it
sbt "runMain app.Main chunk --input data/corpus/large.txt --lines 2000 --yes"

# 3. Train incrementally (one chunk per session)
sbt "runMain app.Main train --input data/corpus/chunks/large-part1.txt --preset quick"
sbt "runMain app.Main train --input data/corpus/chunks/large-part2.txt --preset quick"
sbt "runMain app.Main train --input data/corpus/chunks/large-part3.txt --preset quick"

# 4. Test predictions
sbt "runMain app.Main predict --context 'once upon a time' --topK 5"
```

### Daily Training Routine

```bash
# Morning: 30 min training session
sbt "runMain app.Main train --input data/corpus/today.txt --preset quick"

# Quick test
sbt "runMain app.Main predict --context 'good morning'"

# Continue tomorrow with new data
```

### Training BBC News Dataset

```bash
# 1. Combine all articles
cat data/corpus/bbc/*/*.txt > data/corpus/bbc-combined.txt

# 2. Check size
wc -l data/corpus/bbc-combined.txt

# 3. Chunk into 2000-line files
split -l 2000 data/corpus/bbc-combined.txt data/corpus/chunks/bbc-part-

# 4. Train incrementally
sbt "runMain app.Main train --input data/corpus/chunks/bbc-part-aa --preset balanced"
sbt "runMain app.Main train --input data/corpus/chunks/bbc-part-ab --preset balanced"
# ... continue with remaining parts
```

## Requirements

- Scala 3.3.3 (for language features and compiler)
- SBT 1.10.2 (for project build and dependency management)
- No external ML libraries (pure Scala implementation)

## License

Local project - educational/learning purpose.
