import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sampling.config import SamplingConfig
from src.sampling.samplers import GreedySampler, LocallyTypicalSampler, RandomSampler, TopKSampler, TopPSampler
from src.evaluation.evaluator import EvaluationRunner
from src.data.tokenizer import LMTokenizer
from src.models.gpt import GPT, GPTConfig


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device_arg


def parse_seeds(seed_arg: str):
    return [int(seed.strip()) for seed in seed_arg.split(",") if seed.strip()]


def parse_methods(methods_arg: str):
    if not methods_arg:
        return None
    return {method.strip() for method in methods_arg.split(",") if method.strip()}


def load_generator_model(model_path: str, tokenizer: LMTokenizer, device: str):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model_state_dict = state["model_state_dict"]
        config_dict = state.get("gpt_config")
    elif isinstance(state, dict):
        model_state_dict = state
        config_dict = None
    else:
        raise ValueError(f"Unsupported checkpoint format in: {model_path}")

    config = GPTConfig(**config_dict) if config_dict is not None else GPTConfig(tokenizer.vocab_size)
    model = GPT(config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run language model evaluation")
    parser.add_argument("--model_path", required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Path to the tokenizer JSON file")
    parser.add_argument("--device", default="auto",
                        help="Device to run inference on (auto|cpu|cuda)")
    parser.add_argument("--output_dir", default="evaluation_results",
                        help="Directory for final metrics, plots, and generations.")
    parser.add_argument("--generations_per_prompt", type=int, default=50,
                        help="Number of generations per prompt and seed.")
    parser.add_argument("--seeds", default="42,43,44",
                        help="Comma-separated random seeds.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum generated tokens per sample.")
    parser.add_argument("--perplexity_source", choices=["reference", "local", "none"], default="reference",
                        help="Use distilgpt2 reference perplexity, local GPT perplexity, or skip perplexity.")
    parser.add_argument("--reference_model_name", default="distilgpt2",
                        help="Frozen Hugging Face model used when perplexity_source=reference.")
    parser.add_argument("--perplexity_batch_size", type=int, default=8,
                        help="Batch size for reference-model perplexity.")
    parser.add_argument("--metric_text_field", choices=["continuation", "full_text"], default="continuation",
                        help="Evaluate only generated continuations or prompt plus continuation.")
    parser.add_argument("--methods", default=None,
                        help="Optional comma-separated subset of methods to run.")

    args = parser.parse_args()

    PROMPTS = [
        "Once upon a time",
        "The little cat",
        "In a faraway land",
        "There was a small",
        "One sunny day",
        "The magic forest",
        "A brave little",
        "Long, long ago",
        "The happy dog",
        "In a tiny village",
        "Once there lived",
        "The curious child",
        "Under the bright",
        "In the deep",
        "A friendly monster",
        "The secret garden",
        "On a rainy",
        "The lost kitten",
        "In a magical",
        "The adventurous rabbit"
    ]
    DEVICE = resolve_device(args.device)
    SEEDS = parse_seeds(args.seeds)
    METHOD_FILTER = parse_methods(args.methods)

    tokenizer = LMTokenizer.from_json(args.tokenizer_path)
    model = load_generator_model(args.model_path, tokenizer, DEVICE)

    SAMPLERS_DICT = {
        "greedy": GreedySampler(model, tokenizer, DEVICE),
        "random": RandomSampler(model, tokenizer, DEVICE),
        "temperature": RandomSampler(model, tokenizer, DEVICE),
        "top_k": TopKSampler(model, tokenizer, DEVICE),
        "top_p": TopPSampler(model, tokenizer, DEVICE),
        "repetition_control": RandomSampler(model, tokenizer, DEVICE),
        "locally_typical": LocallyTypicalSampler(model, tokenizer, DEVICE)
    }

    SAMPLING_CONFIGS = {
        "greedy": [SamplingConfig()],
        "random": [SamplingConfig(temperature=1.0)],
        "temperature": [SamplingConfig(temperature=t) for t in [0.5, 0.7, 0.9, 1.0, 1.2]],
        "top_k":  [SamplingConfig(top_k=k) for k in [10, 40, 100]],
        "top_p":  [SamplingConfig(top_p=p) for p in [0.90, 0.95, 0.99]],
        "repetition_control": [
            SamplingConfig(no_repeat_ngram_size=4),
            SamplingConfig(repetition_penalty=1.2),
        ],
        "locally_typical": [SamplingConfig(locally_typical_tau=tau) for tau in [0.1, 0.2, 0.3]]
    }

    if METHOD_FILTER is not None:
        unknown_methods = METHOD_FILTER - set(SAMPLING_CONFIGS.keys())
        if unknown_methods:
            raise ValueError(f"Unknown methods requested: {sorted(unknown_methods)}")
        SAMPLING_CONFIGS = {
            method: configs
            for method, configs in SAMPLING_CONFIGS.items()
            if method in METHOD_FILTER
        }

    runner = EvaluationRunner(
        SAMPLERS_DICT,
        PROMPTS,
        output_dir=args.output_dir,
        device=DEVICE,
        model=model,
        tokenizer=tokenizer,
        reference_model_name=args.reference_model_name,
        perplexity_source=args.perplexity_source,
        perplexity_batch_size=args.perplexity_batch_size,
        metric_text_field=args.metric_text_field,
    )
    results = runner.run_experiments(
        SAMPLING_CONFIGS,
        generations_per_prompt=args.generations_per_prompt,
        seeds=SEEDS,
        max_new_tokens=args.max_new_tokens,
    )
