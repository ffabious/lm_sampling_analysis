import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.loader import build_loaders
from src.data.tokenizer import LMTokenizer
from src.models.gpt import GPTConfig, GPT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_vocab_size(tokenizer_path: str = "checkpoints/tokenizer.json") -> int:
    try:
        tokenizer = LMTokenizer.from_json(tokenizer_path)
    except FileNotFoundError:
        print("Tokenizer file not found. Please run the tokenizer training script first.")
        raise
    return tokenizer.vocab_size


def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")

    candidates = list(checkpoint_root.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint files found in: {checkpoint_root}")

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_model(checkpoint_path: Path, device: str, vocab_size: int):
    payload = torch.load(checkpoint_path, map_location=device)

    checkpoint_meta = {}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model_state_dict = payload["model_state_dict"]
        config_dict = payload.get("gpt_config")
        checkpoint_meta = {
            "epoch": payload.get("epoch"),
            "timestamp": payload.get("timestamp"),
            "hparams": payload.get("hparams"),
            "metrics": payload.get("metrics"),
        }
    elif isinstance(payload, dict):
        model_state_dict = payload
        config_dict = None
    else:
        raise ValueError(f"Unsupported checkpoint format in: {checkpoint_path}")

    if config_dict is None:
        config = GPTConfig(vocab_size=vocab_size)
    else:
        config = GPTConfig(**config_dict)

    model = GPT(config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint_meta, config


def evaluate(model: GPT, test_loader, device: str, max_batches: int | None = None):
    total_loss_sum = 0.0
    total_tokens = 0
    total_correct = 0
    batches_processed = 0

    with torch.inference_mode():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            if max_batches is not None and batches_processed >= max_batches:
                break

            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            out = model(inputs, labels=targets)

            loss = out["loss"]
            logits = out["logits"]

            ignore_index = -100
            valid = targets.ne(ignore_index)
            num_tokens = int(valid.sum().item())

            predictions = torch.argmax(logits, dim=-1)
            correct_tokens = int((predictions.eq(targets) & valid).sum().item())

            total_loss_sum += loss.item() * num_tokens
            total_tokens += num_tokens
            total_correct += correct_tokens
            batches_processed += 1

            running_loss = total_loss_sum / max(total_tokens, 1)
            running_acc = total_correct / max(total_tokens, 1)
            if running_loss < 20:
                running_ppl = math.exp(running_loss)
            else:
                running_ppl = float("inf")

            pbar.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "ppl": f"{running_ppl:.2f}" if math.isfinite(running_ppl) else "inf",
                    "token_acc": f"{running_acc:.4f}",
                }
            )

    avg_loss = total_loss_sum / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    token_accuracy = total_correct / max(total_tokens, 1)

    return {
        "batches_processed": batches_processed,
        "tokens_evaluated": total_tokens,
        "avg_test_loss": avg_loss,
        "test_perplexity": perplexity,
        "token_accuracy": token_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate latest GPT checkpoint on test split.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Specific checkpoint path.")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer.json", help="Tokenizer JSON path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap on test batches.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save evaluation report.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else find_latest_checkpoint(args.checkpoint_dir)
    vocab_size = get_vocab_size(args.tokenizer_path)

    print("=" * 80)
    print("Test run configuration")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Seed: {args.seed}")
    if args.max_batches is not None:
        print(f"Max batches: {args.max_batches}")

    model, checkpoint_meta, config = load_model(checkpoint_path, device, vocab_size)

    print("\nModel / checkpoint info")
    print("-" * 80)
    print(f"GPT config: {vars(config)}")
    if checkpoint_meta:
        print(f"Checkpoint epoch: {checkpoint_meta.get('epoch')}")
        print(f"Checkpoint timestamp: {checkpoint_meta.get('timestamp')}")
        if checkpoint_meta.get("hparams") is not None:
            print(f"Saved hparams: {checkpoint_meta['hparams']}")
        if checkpoint_meta.get("metrics") is not None:
            print(f"Saved train/val metrics: {checkpoint_meta['metrics']}")

    _, _, test_loader = build_loaders(batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    test_size = len(test_loader.dataset)
    print("\nTest data info")
    print("-" * 80)
    print(f"Test examples: {test_size}")
    print(f"Test batches: {len(test_loader)}")

    metrics = evaluate(model, test_loader, device=device, max_batches=args.max_batches)

    print("\nFinal test metrics")
    print("-" * 80)
    print(f"Batches processed: {metrics['batches_processed']}")
    print(f"Tokens evaluated: {metrics['tokens_evaluated']}")
    print(f"Average test loss: {metrics['avg_test_loss']:.6f}")
    if math.isfinite(metrics["test_perplexity"]):
        print(f"Test perplexity: {metrics['test_perplexity']:.6f}")
    else:
        print("Test perplexity: inf")
    print(f"Token accuracy: {metrics['token_accuracy']:.6f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = save_dir / f"test_metrics_{checkpoint_path.stem}_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_path": args.tokenizer_path,
        "device": device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "gpt_config": vars(config),
        "checkpoint_meta": checkpoint_meta,
        "test_size": test_size,
        "metrics": metrics,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved evaluation report: {report_path}")


if __name__ == "__main__":
    main()
