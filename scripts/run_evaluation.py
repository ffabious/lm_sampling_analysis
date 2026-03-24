import argparse
import torch
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run language model evaluation")
    parser.add_argument("--model_path", required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Path to the tokenizer JSON file")
    parser.add_argument("--device", default="auto",
                        help="Device to run inference on (auto|cpu|cuda)")

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

    tokenizer = LMTokenizer.from_json(args.tokenizer_path)

    config = GPTConfig(tokenizer.vocab_size)
    model = GPT(config)

    state = torch.load(args.model_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model_state_dict = state["model_state_dict"]
    elif isinstance(state, dict):
        model_state_dict = state
    else:
        raise ValueError(f"Unsupported checkpoint format in: {args.model_path}")

    model.load_state_dict(model_state_dict)
    model.to(DEVICE)
    model.eval()

    SAMPLERS_DICT = {
        "greedy": GreedySampler(model, tokenizer, DEVICE),
        "random": RandomSampler(model, tokenizer, DEVICE),
        "top_k": TopKSampler(model, tokenizer, DEVICE),
        "top_p": TopPSampler(model, tokenizer, DEVICE),
        "locally_typical": LocallyTypicalSampler(model, tokenizer, DEVICE)
    }

    SAMPLING_CONFIGS = {
        "greedy": [SamplingConfig()],
        "random": [SamplingConfig(temperature=t) for t in [0.7, 1.0, 1.5]],
        "top_k":  [SamplingConfig(top_k=k) for k in [10, 50]],
        "top_p":  [SamplingConfig(top_p=p) for p in [0.9, 0.95]],
        "locally_typical": [SamplingConfig(locally_typical_tau=tau) for tau in [0.2, 0.9, 1.5]]
    }

    runner = EvaluationRunner(SAMPLERS_DICT, PROMPTS, device=DEVICE, model=model, tokenizer=tokenizer)
    results = runner.run_experiments(
        SAMPLING_CONFIGS, generations_per_prompt=10, seeds=[42,])
