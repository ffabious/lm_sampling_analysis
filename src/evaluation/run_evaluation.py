import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.sampling.samplers import Sampler, GreedySampler, RandomSampler, SamplingConfig
from src.evaluation.metrics import EvaluationPipeline
from src.data.tokenizer import LMTokenizer
from src.models.gpt import GPT, GPTConfig


class EvaluationRunner:
    """Run evaluation experiments for different sampling methods."""
    
    def __init__(
        self,
        model,
        tokenizer,
        prompts: List[str],
        output_dir: str = "evaluation_results",
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize samplers
        self.sampler = Sampler(model, tokenizer, device)
        self.greedy_sampler = GreedySampler(model, tokenizer, device)
        self.random_sampler = RandomSampler(model, tokenizer, device)
        
        # Initialize evaluation pipeline
        self.evaluator = EvaluationPipeline(device=device)
    
    def run_experiments(
        self,
        sampling_configs: Dict[str, List[SamplingConfig]],
        generations_per_prompt: int = 50,
        seeds: List[int] = [42, 43, 44],
        max_new_tokens: int = 256
    ):
        """
        Run experiments for all sampling configurations.
        
        Args:
            sampling_configs: Dictionary mapping method name to list of configs
            generations_per_prompt: Number of generations per prompt
            seeds: Random seeds for reproducibility
            max_new_tokens: Maximum tokens to generate
        """
        results = {}
        
        for method_name, configs in sampling_configs.items():
            print(f"\nEvaluating {method_name}...")
            method_results = []
            
            for config_idx, config in enumerate(configs):
                print(f"  Config {config_idx + 1}/{len(configs)}: {config}")
                
                for seed in seeds:
                    # Generate texts
                    all_generated_texts = []
                    
                    for prompt in tqdm(self.prompts, desc=f"    Seed {seed}"):
                        for _ in range(generations_per_prompt):
                            if method_name == "greedy":
                                text, _ = self.greedy_sampler.generate(
                                    prompt, max_new_tokens=max_new_tokens, seed=seed
                                )
                            elif method_name == "random":
                                text, _ = self.random_sampler.generate(
                                    prompt, config, max_new_tokens=max_new_tokens, seed=seed
                                )
                            else:
                                text, _ = self.sampler.generate(
                                    prompt, config, max_new_tokens=max_new_tokens, seed=seed
                                )
                            all_generated_texts.append(text)
                    
                    # Evaluate
                    metrics = self.evaluator.evaluate(all_generated_texts)
                    
                    method_results.append({
                        "method": method_name,
                        "config": {
                            "temperature": config.temperature,
                            "top_k": config.top_k,
                            "top_p": config.top_p,
                            "repetition_penalty": config.repetition_penalty,
                            "no_repeat_ngram_size": config.no_repeat_ngram_size,
                            "locally_typical_tau": config.locally_typical_tau,
                        },
                        "seed": seed,
                        "metrics": metrics,
                        "num_samples": len(all_generated_texts)
                    })
            
            results[method_name] = method_results
        
        # Save results
        self._save_results(results)
        
        # Generate plots
        self._generate_plots(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        # Convert numpy/torch types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            return obj
        
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, default=convert_to_serializable, indent=2)
    
    def _generate_plots(self, results: Dict):
        """Generate evaluation plots."""
        # Prepare data for plotting
        plot_data = []
        
        for method_name, method_results in results.items():
            for result in method_results:
                plot_data.append({
                    "method": method_name,
                    "config": str(result["config"]),
                    "perplexity": result["metrics"]["perplexity"],
                    "self_bleu": result["metrics"]["self_bleu"],
                    "repetition_4": result["metrics"]["repetition_4"],
                    "distinct_2": result["metrics"]["distinct_2"],
                    "seed": result["seed"]
                })
        
        # Quality vs Diversity plot
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")
        
        for method_name in results.keys():
            method_points = [d for d in plot_data if d["method"] == method_name]
            if method_points:
                x = [d["self_bleu"] for d in method_points]
                y = [d["perplexity"] for d in method_points]
                plt.scatter(x, y, label=method_name, alpha=0.6, s=100)
        
        plt.xlabel("Self-BLEU (lower = more diverse)", fontsize=12)
        plt.ylabel("Perplexity (lower = more natural)", fontsize=12)
        plt.title("Quality vs Diversity Trade-off", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_vs_diversity.png", dpi=150)
        plt.close()
        
        # Repetition plot
        plt.figure(figsize=(12, 6))
        
        methods = list(results.keys())
        x_pos = np.arange(len(methods))
        width = 0.35
        
        repetition_means = []
        repetition_stds = []
        distinct_means = []
        distinct_stds = []
        
        for method in methods:
            method_results = [d for d in plot_data if d["method"] == method]
            repetition = [d["repetition_4"] for d in method_results]
            distinct = [d["distinct_2"] for d in method_results]
            
            repetition_means.append(np.mean(repetition))
            repetition_stds.append(np.std(repetition))
            distinct_means.append(np.mean(distinct))
            distinct_stds.append(np.std(distinct))
        
        plt.subplot(1, 2, 1)
        plt.bar(x_pos, repetition_means, yerr=repetition_stds, capsize=5)
        plt.xticks(x_pos, methods, rotation=45, ha='right')
        plt.ylabel("Repetition-4 (lower is better)")
        plt.title("Repetition Rate")
        
        plt.subplot(1, 2, 2)
        plt.bar(x_pos, distinct_means, yerr=distinct_stds, capsize=5)
        plt.xticks(x_pos, methods, rotation=45, ha='right')
        plt.ylabel("Distinct-2 (higher is better)")
        plt.title("Vocabulary Richness")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "repetition_and_diversity.png", dpi=150)
        plt.close()
        
        print(f"\nPlots saved to {self.output_dir}")


# Example usage script
if __name__ == "__main__":
    # Define prompts
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
    
    # Define sampling configurations
    SAMPLING_CONFIGS = {
        "greedy": [SamplingConfig()],  # Greedy ignores config
        
        "random": [
            SamplingConfig(temperature=t) 
            for t in [0.5, 0.7, 0.9, 1.0, 1.2]
        ],
        
        "temperature": [
            SamplingConfig(temperature=t)
            for t in [0.5, 0.7, 0.9, 1.0, 1.2]
        ],
        
        "top_k": [
            SamplingConfig(top_k=k, temperature=0.9)
            for k in [10, 40, 100]
        ],
        
        "top_p": [
            SamplingConfig(top_p=p, temperature=0.9)
            for p in [0.90, 0.95, 0.99]
        ],
        
        "locally_typical": [
            SamplingConfig(locally_typical_tau=tau, temperature=0.9)
            for tau in [0.1, 0.2, 0.3]
        ],
        
        "repetition_penalty": [
            SamplingConfig(repetition_penalty=1.2, temperature=0.9)
        ],
        
        "no_repeat_ngram": [
            SamplingConfig(no_repeat_ngram_size=4, temperature=0.9)
        ]
    }

    PATH_TO_MODEL = ""

    PATH_TO_TOKENIZER = ""

    config = GPTConfig()
    model = GPT(config)
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    tokenizer = LMTokenizer.from_json(PATH_TO_TOKENIZER)

    runner = EvaluationRunner(model, tokenizer, PROMPTS, device="cpu")
    results = runner.run_experiments(SAMPLING_CONFIGS)
