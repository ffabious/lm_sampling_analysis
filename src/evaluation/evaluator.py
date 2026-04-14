import csv
import json
import time
from dataclasses import asdict, is_dataclass
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


from src.evaluation.metrics import EvaluationPipeline


METRIC_KEYS = ("perplexity", "self_bleu", "repetition_4", "distinct_2")


class EvaluationRunner:
    """Run evaluation experiments mapping algorithm names to initialized samplers."""

    def __init__(
        self,
        samplers_dict,
        prompts: List[str],
        output_dir: str = "evaluation_results",
        device: str = "cpu",
        model=None,
        tokenizer=None,
        reference_model_name: str = "distilgpt2",
        perplexity_source: str = "reference",
        perplexity_batch_size: int = 8,
        metric_text_field: str = "continuation",
    ):
        if metric_text_field not in {"continuation", "full_text"}:
            raise ValueError("metric_text_field must be continuation or full_text")

        self.samplers = samplers_dict
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_text_field = metric_text_field
        self.evaluator = EvaluationPipeline(
            device=device,
            model=model,
            tokenizer=tokenizer,
            reference_model_name=reference_model_name,
            perplexity_source=perplexity_source,
            perplexity_batch_size=perplexity_batch_size,
        )
        self.perplexity_source = perplexity_source
        self.reference_model_name = reference_model_name

    def run_experiments(
        self,
        sampling_configs: dict,
        generations_per_prompt: int = 50,
        seeds: List[int] = None,
        max_new_tokens: int = 256,
        save_generations: bool = True,
    ):
        if seeds is None:
            seeds = [42, 43, 44]

        generation_path = self.output_dir / "generations.jsonl"
        if save_generations:
            generation_path.write_text("", encoding="utf-8")

        seed_results = []
        for method_name, configs in sampling_configs.items():
            print(f"\nEvaluating {method_name}...")
            sampler = self.samplers.get(method_name)
            if not sampler:
                continue

            for config_index, config in enumerate(configs):
                print(f"  Config {config_index + 1}: {config}")
                for seed in seeds:
                    all_texts = []
                    generation_records = []

                    for prompt_index, prompt in enumerate(tqdm(self.prompts, desc=f"    Seed {seed}")):
                        for sample_index in range(generations_per_prompt):
                            generation_seed = self._generation_seed(
                                seed,
                                prompt_index,
                                sample_index,
                                generations_per_prompt,
                            )
                            details = sampler.generate_details(
                                prompt,
                                config,
                                max_new_tokens=max_new_tokens,
                                seed=generation_seed,
                            )

                            metric_text = details[self.metric_text_field]
                            all_texts.append(metric_text)
                            generation_records.append({
                                "method": method_name,
                                "config_index": config_index,
                                "config": self._config_to_dict(config),
                                "config_label": self._config_label(config),
                                "seed": seed,
                                "generation_seed": generation_seed,
                                "prompt_id": prompt_index,
                                "prompt": prompt,
                                "sample_id": sample_index,
                                "full_text": details["full_text"],
                                "continuation": details["continuation"],
                                "metric_text_field": self.metric_text_field,
                                "new_token_count": details["new_token_count"],
                                "stopped_on_eos": details["stopped_on_eos"],
                            })

                    if save_generations:
                        self._append_jsonl(generation_path, generation_records)

                    metrics = self.evaluator.evaluate(
                        all_texts,
                        compute_perplexity=self.perplexity_source != "none",
                    )

                    seed_results.append({
                        "method": method_name,
                        "config_index": config_index,
                        "config": self._config_to_dict(config),
                        "config_label": self._config_label(config),
                        "seed": seed,
                        "metrics": metrics,
                        "num_samples": len(all_texts),
                        "mean_new_tokens": float(np.mean([
                            record["new_token_count"] for record in generation_records
                        ])),
                        "eos_rate": float(np.mean([
                            record["stopped_on_eos"] for record in generation_records
                        ])),
                    })

        summary = self._summarize_seed_results(seed_results)
        results = {
            "metadata": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_prompts": len(self.prompts),
                "prompts": self.prompts,
                "generations_per_prompt": generations_per_prompt,
                "seeds": seeds,
                "max_new_tokens": max_new_tokens,
                "metric_text_field": self.metric_text_field,
                "perplexity_source": self.perplexity_source,
                "reference_model_name": (
                    self.reference_model_name if self.perplexity_source == "reference" else None
                ),
                "generation_file": str(generation_path) if save_generations else None,
            },
            "seed_results": seed_results,
            "summary": summary,
        }
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
        self._save_summary_csv(results)

    def _save_summary_csv(self, results: Dict):
        fieldnames = [
            "method",
            "config_index",
            "config_label",
            "num_seeds",
            "seeds",
            "num_samples_per_seed",
            "perplexity_mean",
            "perplexity_std",
            "self_bleu_mean",
            "self_bleu_std",
            "repetition_4_mean",
            "repetition_4_std",
            "distinct_2_mean",
            "distinct_2_std",
            "mean_new_tokens_mean",
            "mean_new_tokens_std",
            "eos_rate_mean",
            "eos_rate_std",
        ]
        with open(self.output_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results["summary"]:
                metrics = row["metrics"]
                writer.writerow({
                    "method": row["method"],
                    "config_index": row["config_index"],
                    "config_label": row["config_label"],
                    "num_seeds": row["num_seeds"],
                    "seeds": ",".join(str(seed) for seed in row["seeds"]),
                    "num_samples_per_seed": ",".join(
                        str(num_samples) for num_samples in row["num_samples_per_seed"]
                    ),
                    "perplexity_mean": metrics.get("perplexity", {}).get("mean"),
                    "perplexity_std": metrics.get("perplexity", {}).get("std"),
                    "self_bleu_mean": metrics["self_bleu"]["mean"],
                    "self_bleu_std": metrics["self_bleu"]["std"],
                    "repetition_4_mean": metrics["repetition_4"]["mean"],
                    "repetition_4_std": metrics["repetition_4"]["std"],
                    "distinct_2_mean": metrics["distinct_2"]["mean"],
                    "distinct_2_std": metrics["distinct_2"]["std"],
                    "mean_new_tokens_mean": row["mean_new_tokens"]["mean"],
                    "mean_new_tokens_std": row["mean_new_tokens"]["std"],
                    "eos_rate_mean": row["eos_rate"]["mean"],
                    "eos_rate_std": row["eos_rate"]["std"],
                })

    def _generate_plots(self, results: Dict):
        plot_data = [
            {
                "method": row["method"],
                "config": row.get("config", {}),
                "config_label": row["config_label"],
                "perplexity_mean": row["metrics"].get("perplexity", {}).get("mean"),
                "perplexity_std": row["metrics"].get("perplexity", {}).get("std"),
                "self_bleu_mean": row["metrics"]["self_bleu"]["mean"],
                "self_bleu_std": row["metrics"]["self_bleu"]["std"],
                "repetition_4_mean": row["metrics"]["repetition_4"]["mean"],
                "repetition_4_std": row["metrics"]["repetition_4"]["std"],
                "distinct_2_mean": row["metrics"]["distinct_2"]["mean"],
                "distinct_2_std": row["metrics"]["distinct_2"]["std"],
            }
            for row in results["summary"]
        ]

        # Quality vs diversity plot
        if self.perplexity_source != "none":
            plt.figure(figsize=(10, 8))
            sns.set_style("whitegrid")

            all_x = []
            all_y = []

            for method_name in sorted({d["method"] for d in plot_data}):
                method_points = [
                    d for d in plot_data
                    if d["method"] == method_name and d["perplexity_mean"] is not None
                ]
                if method_points:
                    x = [d["self_bleu_mean"] for d in method_points]
                    y = [d["perplexity_mean"] for d in method_points]
                    xerr = [d["self_bleu_std"] for d in method_points]
                    yerr = [d["perplexity_std"] for d in method_points]
                    all_x.extend(x)
                    all_y.extend(y)
                    plt.errorbar(
                        x,
                        y,
                        xerr=xerr,
                        yerr=yerr,
                        fmt="o",
                        capsize=3,
                        label=self._method_label(method_name),
                        alpha=0.75,
                    )

            fit = self._fit_hyperbolic_curve(np.array(all_x, dtype=float), np.array(all_y, dtype=float))
            if fit is not None:
                x_min = float(np.min(all_x))
                x_max = float(np.max(all_x))
                x_fit = np.linspace(x_min, x_max, 400)
                y_fit = fit["a"] / (x_fit + fit["b"]) + fit["c"]
                plt.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    color="black",
                    linewidth=1.8,
                    alpha=0.9,
                    label=f"Hyperbolic fit (R^2={fit['r2']:.3f})",
                )
                b_sign = "+" if fit["b"] >= 0 else "-"
                c_sign = "+" if fit["c"] >= 0 else "-"
                equation = (
                    f"y = {fit['a']:.2f} / (x {b_sign} {abs(fit['b']):.3f}) "
                    f"{c_sign} {abs(fit['c']):.2f}"
                )
                plt.text(
                    0.02,
                    0.98,
                    equation,
                    transform=plt.gca().transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7},
                )

            plt.xlabel("Self-BLEU (lower = more diverse)", fontsize=12)
            plt.ylabel("Perplexity (lower = more natural)", fontsize=12)
            plt.title("Quality vs Diversity Trade-off", fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "quality_vs_diversity.png", dpi=150)
            plt.close()

        # Repetition plot
        if not plot_data:
            return

        plt.figure(figsize=(16, 6))

        labels = [self._plot_label(d) for d in plot_data]
        x_pos = np.arange(len(labels))

        repetition_means = [d["repetition_4_mean"] for d in plot_data]
        repetition_stds = [d["repetition_4_std"] for d in plot_data]
        distinct_means = [d["distinct_2_mean"] for d in plot_data]
        distinct_stds = [d["distinct_2_std"] for d in plot_data]

        plt.subplot(1, 2, 1)
        plt.bar(x_pos, repetition_means, yerr=repetition_stds, capsize=5)
        plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=9)
        plt.ylabel("Repetition-4 (lower is better)")
        plt.title("Repetition rate")

        plt.subplot(1, 2, 2)
        plt.bar(x_pos, distinct_means, yerr=distinct_stds, capsize=5)
        plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=9)
        plt.ylabel("Distinct-2 (higher is better)")
        plt.title("Vocabulary richness")

        plt.tight_layout()
        plt.savefig(self.output_dir / "repetition_and_diversity.png", dpi=150)
        plt.close()

        print(f"\nPlots saved to {self.output_dir}")

    @staticmethod
    def _append_jsonl(path: Path, records: List[Dict]):
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _config_to_dict(config):
        if is_dataclass(config):
            return asdict(config)
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)
        return {"value": str(config)}

    @staticmethod
    def _config_label(config) -> str:
        config_dict = EvaluationRunner._config_to_dict(config)
        active = [
            f"{key}={value}"
            for key, value in config_dict.items()
            if value is not None and not (key == "temperature" and value == 1.0)
        ]
        return ", ".join(active) if active else "default"

    @staticmethod
    def _method_label(method: str) -> str:
        labels = {
            "greedy": "Greedy",
            "random": "Random",
            "temperature": "Temperature",
            "top_k": "Top-k",
            "top_p": "Top-p",
            "repetition_control": "Repetition control",
            "locally_typical": "Locally typical",
        }
        return labels.get(method, method.replace("_", " ").title())

    @staticmethod
    def _plot_label(row: Dict) -> str:
        method = row["method"]
        config = row.get("config") or {}

        if method == "greedy":
            return "Greedy"
        if method == "random":
            return "Random"
        if method == "temperature":
            return f"Temp {EvaluationRunner._format_config_value(config.get('temperature', 1.0))}"
        if method == "top_k" and config.get("top_k") is not None:
            return f"Top-k {EvaluationRunner._format_config_value(config['top_k'])}"
        if method == "top_p" and config.get("top_p") is not None:
            return f"Top-p {EvaluationRunner._format_config_value(config['top_p'])}"
        if method == "locally_typical" and config.get("locally_typical_tau") is not None:
            return f"Typical {EvaluationRunner._format_config_value(config['locally_typical_tau'])}"
        if method == "repetition_control":
            if config.get("no_repeat_ngram_size") is not None:
                return f"No repeat {EvaluationRunner._format_config_value(config['no_repeat_ngram_size'])}"
            if config.get("repetition_penalty") is not None:
                return f"Rep pen {EvaluationRunner._format_config_value(config['repetition_penalty'])}"

        config_label = row.get("config_label", "default")
        if config_label == "default":
            return EvaluationRunner._method_label(method)
        return EvaluationRunner._short_config_label(config_label)

    @staticmethod
    def _short_config_label(config_label: str) -> str:
        return (
            config_label
            .replace("locally_typical_tau", "tau")
            .replace("no_repeat_ngram_size", "no repeat")
            .replace("repetition_penalty", "rep pen")
            .replace("temperature", "temp")
            .replace("_", " ")
        )

    @staticmethod
    def _format_config_value(value) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    @staticmethod
    def _generation_seed(
        seed: int,
        prompt_index: int,
        sample_index: int,
        generations_per_prompt: int,
    ) -> int:
        return seed * 1_000_000 + prompt_index * generations_per_prompt + sample_index

    @staticmethod
    def _fit_hyperbolic_curve(x: np.ndarray, y: np.ndarray):
        """Fit y = a / (x + b) + c by grid-searching b and solving (a, c) with least squares."""
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]

        if x.size < 3:
            return None

        min_x = float(np.min(x))
        max_x = float(np.max(x))
        if np.isclose(min_x, max_x):
            return None

        lower_b = -min_x + 1e-4
        upper_b = max(2.0, 10.0 * (max_x - min_x))
        b_candidates = np.linspace(lower_b, upper_b, 800)

        best = None
        for b in b_candidates:
            denom = x + b
            if np.any(denom <= 1e-9):
                continue

            design = np.column_stack((1.0 / denom, np.ones_like(denom)))
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            a, c = float(coeffs[0]), float(coeffs[1])
            y_pred = design @ coeffs
            sse = float(np.sum((y - y_pred) ** 2))

            if not np.isfinite(sse):
                continue
            if best is None or sse < best["sse"]:
                best = {"sse": sse, "a": a, "b": float(b), "c": c}

        if best is None:
            return None

        sst = float(np.sum((y - np.mean(y)) ** 2))
        best["r2"] = 1.0 - best["sse"] / sst if sst > 0 else 1.0
        return best

    @staticmethod
    def _summarize_seed_results(seed_results: List[Dict]) -> List[Dict]:
        grouped = {}
        for result in seed_results:
            key = (result["method"], result["config_index"])
            grouped.setdefault(key, []).append(result)

        summary = []
        for (method, config_index), rows in grouped.items():
            first = rows[0]
            metrics = {}
            for metric in METRIC_KEYS:
                values = [
                    row["metrics"][metric]
                    for row in rows
                    if metric in row["metrics"] and np.isfinite(row["metrics"][metric])
                ]
                if not values:
                    continue
                metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": [float(value) for value in values],
                }

            summary.append({
                "method": method,
                "config_index": config_index,
                "config": first["config"],
                "config_label": first["config_label"],
                "num_seeds": len(rows),
                "seeds": [row["seed"] for row in rows],
                "num_samples_per_seed": [row["num_samples"] for row in rows],
                "mean_new_tokens": {
                    "mean": float(np.mean([row["mean_new_tokens"] for row in rows])),
                    "std": float(np.std([row["mean_new_tokens"] for row in rows], ddof=1))
                    if len(rows) > 1 else 0.0,
                },
                "eos_rate": {
                    "mean": float(np.mean([row["eos_rate"] for row in rows])),
                    "std": float(np.std([row["eos_rate"] for row in rows], ddof=1))
                    if len(rows) > 1 else 0.0,
                },
                "metrics": metrics,
            })

        return sorted(summary, key=lambda row: (row["method"], row["config_index"]))
