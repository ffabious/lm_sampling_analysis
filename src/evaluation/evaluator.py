import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


from src.evaluation.metrics import EvaluationPipeline


class EvaluationRunner:
    """Run evaluation experiments mapping algorithm names to initialized samplers."""

    def __init__(self, samplers_dict, prompts: List[str], output_dir: str = "evaluation_results", device: str = "cpu", model=None, tokenizer=None):
        self.samplers = samplers_dict
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = EvaluationPipeline(device=device, model=model, tokenizer=tokenizer)

    def run_experiments(self, sampling_configs: dict, generations_per_prompt: int = 5, seeds: List[int] = [42], max_new_tokens: int = 256):
        results = {}
        for method_name, configs in sampling_configs.items():
            print(f"\nEvaluating {method_name}...")
            sampler = self.samplers.get(method_name)
            if not sampler:
                continue

            method_results = []
            for idx, config in enumerate(configs):
                print(f"  Config {idx + 1}: {config}")
                for seed in seeds:
                    all_texts = [
                        sampler.generate(
                            prompt, config, max_new_tokens, seed=seed)[0]
                        for prompt in tqdm(self.prompts, desc=f"    Seed {seed}")
                        for _ in range(generations_per_prompt)
                    ]

                    method_results.append({
                        "method": method_name,
                        "config": str(config),
                        "seed": seed,
                        "metrics": self.evaluator.evaluate(all_texts, compute_perplexity=True),
                        "num_samples": len(all_texts)
                    })
            results[method_name] = method_results

        self._generate_plots(results)
        self._save_results(results)
        return results

    def _save_results(self, results: Dict):
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

        # Quality vs diversity plot
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")

        for method_name in results.keys():
            method_points = [
                d for d in plot_data if d["method"] == method_name]
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
        plt.title("Repetition rate")

        plt.subplot(1, 2, 2)
        plt.bar(x_pos, distinct_means, yerr=distinct_stds, capsize=5)
        plt.xticks(x_pos, methods, rotation=45, ha='right')
        plt.ylabel("Distinct-2 (higher is better)")
        plt.title("Vocabulary richness")

        plt.tight_layout()
        plt.savefig(self.output_dir / "repetition_and_diversity.png", dpi=150)
        plt.close()

        print(f"\nPlots saved to {self.output_dir}")
