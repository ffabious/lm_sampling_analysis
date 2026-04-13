import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from src.sampling.config import SamplingConfig


class BaseSampler:
    """Base sampler containing the core autoregressive generation loop."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self._generation_calls = 0

    def _process_logits(self, logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
        """To be overridden by specialized subclasses to apply specific filtering."""
        return logits

    def _select_token(self, probs: torch.Tensor) -> torch.Tensor:
        """To be overridden by specialized subclasses (e.g., greedy vs multinomial)."""
        return torch.multinomial(probs, num_samples=1)[0]

    @torch.no_grad()
    def generate_details(
        self,
        prompt: str,
        config: SamplingConfig,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Dict:

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._generation_calls += 1

        config.validate()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_eos=False)

        input_ids = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device)
        prompt_token_count = input_ids.size(1)

        max_seq_len = getattr(self.model.config, "max_seq_len", None)
        if max_seq_len is not None and input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            prompt_token_count = input_ids.size(1)

        generated = input_ids.clone()
        past_key_values = None
        all_logits = []
        stopped_on_eos = False

        # Track generated n-grams for repetition control
        generated_ngrams = set()
        if config.no_repeat_ngram_size is not None:
            generated_ngrams = self._build_ngram_set(
                generated[0].tolist(), config.no_repeat_ngram_size)

        # Autoregressive loop
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                cached_len = past_key_values[0][0].size(2)
                if max_seq_len is not None and cached_len >= max_seq_len:
                    past_key_values = None
                    model_input = generated[:, -max_seq_len:]
                else:
                    # Only pass the last token when we have cached KV
                    model_input = generated[:, -1:]
            else:
                if max_seq_len is not None and generated.size(1) > max_seq_len:
                    model_input = generated[:, -max_seq_len:]
                else:
                    model_input = generated
            
            outputs = self.model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = outputs["logits"][0, -1, :]  # Shape: (vocab_size,)
            past_key_values = outputs.get("past_key_values")
            all_logits.append(logits.cpu())

            next_token_logits = logits.clone()

            # Apply base constraints
            if config.repetition_penalty is not None:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated[0], config.repetition_penalty
                )

            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature

            if config.no_repeat_ngram_size is not None:
                next_token_logits = self._apply_no_repeat_ngrams(
                    next_token_logits, generated[0], generated_ngrams, config.no_repeat_ngram_size
                )

            # Filter logits
            next_token_logits = self._process_logits(next_token_logits, config)

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                fallback_logits = logits.clone()
                if config.temperature != 1.0:
                    fallback_logits = fallback_logits / config.temperature
                probs = F.softmax(fallback_logits, dim=-1)
                if not torch.isfinite(probs).all() or probs.sum() <= 0:
                    probs = torch.full_like(probs, 1.0 / probs.numel())
            next_token = self._select_token(probs)

            # Update states
            if config.no_repeat_ngram_size is not None:
                new_sequence = torch.cat(
                    [generated[0], next_token.unsqueeze(0)])
                self._update_ngram_set(
                    generated_ngrams, new_sequence.tolist(), config.no_repeat_ngram_size)

            generated = torch.cat(
                [generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # Check EOS
            if next_token.item() == self.tokenizer.special_tokens.get("<EOS>", -1):
                stopped_on_eos = True
                break

        all_token_ids = generated[0].tolist()
        continuation_token_ids = all_token_ids[prompt_token_count:]
        output_token_ids = self._strip_output_special_tokens(all_token_ids)
        output_continuation_ids = self._strip_output_special_tokens(continuation_token_ids)

        return {
            "full_text": self.tokenizer.decode(output_token_ids),
            "continuation": self.tokenizer.decode(output_continuation_ids),
            "prompt_token_count": prompt_token_count,
            "new_token_count": len(continuation_token_ids),
            "stopped_on_eos": stopped_on_eos,
            "all_token_ids": all_token_ids,
            "continuation_token_ids": continuation_token_ids,
            "logits": all_logits,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: SamplingConfig,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[torch.Tensor]]:
        details = self.generate_details(prompt, config, max_new_tokens=max_new_tokens, seed=seed)
        return details["full_text"], details["logits"]

    def _strip_output_special_tokens(self, token_ids: List[int]) -> List[int]:
        excluded = {
            token_id
            for token, token_id in self.tokenizer.special_tokens.items()
            if token in {"<EOS>", "<PAD>"}
        }
        if not excluded:
            return token_ids
        return [token_id for token_id in token_ids if token_id not in excluded]



    # --- Utility Methods ---
    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        if penalty == 1.0: return logits
        unique_tokens = torch.unique(generated_ids)
        for token in unique_tokens:
            logits[token] = logits[token] * penalty if logits[token] < 0 else logits[token] / penalty
        return logits

    def _build_ngram_set(self, token_ids: List[int], ngram_size: int) -> Set[Tuple[int]]:
        return {tuple(token_ids[i:i + ngram_size]) for i in range(len(token_ids) - ngram_size + 1)} if len(token_ids) >= ngram_size else set()

    def _update_ngram_set(self, ngram_set: Set[Tuple[int]], token_ids: List[int], ngram_size: int):
        if len(token_ids) >= ngram_size:
            ngram_set.add(tuple(token_ids[-ngram_size:]))

    def _apply_no_repeat_ngrams(self, logits: torch.Tensor, generated_ids: torch.Tensor, ngram_set: set, ngram_size: int) -> torch.Tensor:
        if len(generated_ids) < ngram_size - 1: return logits
        context = generated_ids[-(ngram_size - 1):].tolist()
        for token_id in range(logits.size(-1)):
            if tuple(context + [token_id]) in ngram_set:
                logits[token_id] = -float('Inf')
        return logits


class GreedySampler(BaseSampler):
    """Always picks the token with the highest probability."""
    def _select_token(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(probs, dim=-1)


class RandomSampler(BaseSampler):
    """Pure random sampling using only model temperature."""
    def _process_logits(self, logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
        return logits # Applies no filtering, falls through to multinomial


class TopKSampler(BaseSampler):
    """Truncates vocabulary to the top K most likely tokens."""
    def _process_logits(self, logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
        if config.top_k is None or config.top_k <= 0: 
            return logits

        top_k_values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
        logits[logits < top_k_values[-1]] = -float('Inf')
        return logits


class TopPSampler(BaseSampler):
    """Nucleus sampling: truncates down to smallest cumulative probability mass P."""
    def _process_logits(self, logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
        if config.top_p is None or config.top_p <= 0 or config.top_p > 1:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative prob above p
        sorted_indices_to_remove = cumulative_probs > config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        return logits


class LocallyTypicalSampler(BaseSampler):
    """Filters logits using locally typical sampling."""
    def _process_logits(self, logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
        tau = config.locally_typical_tau
        if tau is None:
            return logits
        if not (0.0 < tau <= 1.0):
            raise ValueError("locally_typical_tau must be in (0, 1].")

        if torch.all(torch.isneginf(logits)):
            return logits

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()

        abs_diff = torch.abs(-log_probs - entropy)

        sorted_diff, sorted_indices = torch.sort(abs_diff, descending=False)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        keep_sorted = cumulative_probs < tau
        keep_sorted[0] = True

        cutoff = torch.searchsorted(cumulative_probs, torch.tensor(tau, device=logits.device))
        cutoff = min(cutoff.item(), logits.size(-1) - 1)
        keep_sorted[cutoff] = True

        keep_mask = torch.zeros_like(logits, dtype=torch.bool)
        keep_mask[sorted_indices[keep_sorted]] = True

        filtered_logits = logits.masked_fill(~keep_mask, -float("inf"))
        return filtered_logits
