import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Set
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
    def generate(
        self,
        prompt: str,
        config: SamplingConfig,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[torch.Tensor]]:

        if seed is not None:
            # Avoid repeating identical samples when the caller reuses the same seed.
            effective_seed = seed + self._generation_calls
            torch.manual_seed(effective_seed)
            np.random.seed(effective_seed)
        self._generation_calls += 1

        config.validate()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_eos=False)

        input_ids = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device)

        max_seq_len = getattr(self.model.config, "max_seq_len", None)
        if max_seq_len is not None and input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]

        generated = input_ids.clone()
        past_key_values = None
        all_logits = []

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
                break

        return self.tokenizer.decode(generated[0].tolist()), all_logits



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
    """Filters out tokens not consistent with the typical set (entropy based)."""
    def _process_logits(self, logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
        if config.locally_typical_tau is None: return logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs)
        
        abs_diff = torch.abs(-log_probs - entropy)
        mask = abs_diff <= config.locally_typical_tau
        
        if not mask.any(): 
            return logits

        return torch.where(mask, logits, torch.full_like(logits, -float('Inf')))
