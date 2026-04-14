# Language Model Sampling Analysis

Train a small language model (GPT) on TinyStories and compare decoding strategies with
perplexity, Self-BLEU, repetition, and Distinct-2 metrics.

## Setup

This project uses Poetry and Python 3.10-3.12.

```bash
pip install poetry
```

```bash
poetry install --no-root
```

All commands below should be run from the repository root.

## End-to-End Reproduction

### 1. Train the tokenizer

```bash
poetry run python scripts/train_tokenizer.py
```

Output:

```text
checkpoints/tokenizer.json
```

### 2. Preprocess TinyStories

```bash
poetry run python scripts/preprocess_data.py
```

Output:

```text
data/processed/train/
data/processed/validation/
data/processed/test/
data/processed/metadata.json
```

The validation split from TinyStories is split in half locally: one half is
saved as validation and the other as test.

### 3. Train the GPT model

```bash
poetry run python scripts/train_model.py \
  --device cpu \
  --batch_size 32 \
  --epochs 10 \
  --lr 1e-4 \
  --seed 42
```

Use `--device cuda` or `--device mps` instead if your PyTorch install supports
that device.

Output:

```text
checkpoints/gpt_checkpoint_<timestamp>.pth
```

### 4. Evaluate the trained model on the test split

```bash
poetry run python scripts/test_model.py \
  --checkpoint_path checkpoints/gpt_checkpoint_<timestamp>.pth \
  --tokenizer_path checkpoints/tokenizer.json \
  --device auto \
  --seed 42
```

Output:

```text
results/test_metrics_<checkpoint>_<timestamp>.json
```

### 5. Evaluate sampling approaches

Run the full sampling comparison:

```bash
MPLBACKEND=Agg poetry run python scripts/run_evaluation.py \
  --model_path checkpoints/gpt_checkpoint_<timestamp>.pth \
  --tokenizer_path checkpoints/tokenizer.json \
  --output_dir evaluation_results \
  --generations_per_prompt 50 \
  --seeds 42,43,44 \
  --max_new_tokens 256 \
  --perplexity_source reference
```

Output:

```text
evaluation_results/generations.jsonl
evaluation_results/results.json
evaluation_results/summary.csv
evaluation_results/quality_vs_diversity.png
evaluation_results/repetition_and_diversity.png
```

## Sampling Methods

The evaluation script supports these method names:

```text
greedy
random
temperature
top_k
top_p
repetition_control
locally_typical
```

To evaluate only one method, use `--methods`. For example, rerun only locally
typical sampling:

```bash
MPLBACKEND=Agg poetry run python scripts/run_evaluation.py \
  --model_path checkpoints/gpt_checkpoint_<timestamp>.pth \
  --tokenizer_path checkpoints/tokenizer.json \
  --methods locally_typical \
  --output_dir evaluation_results_locally_typical \
  --generations_per_prompt 50 \
  --seeds 42,43,44 \
  --max_new_tokens 256 \
  --perplexity_source reference
```

Use a separate `--output_dir` when rerunning one method so older full-run
results are not overwritten.

## Faster Smoke Run

For a quick correctness check:

```bash
MPLBACKEND=Agg poetry run python scripts/run_evaluation.py \
  --model_path checkpoints/gpt_checkpoint_<timestamp>.pth \
  --tokenizer_path checkpoints/tokenizer.json \
  --methods locally_typical \
  --output_dir evaluation_results_smoke \
  --generations_per_prompt 5 \
  --seeds 42 \
  --max_new_tokens 128 \
  --perplexity_source none
```