import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    locally_typical_tau: Optional[float] = None
    
    def validate(self):
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")
        
        if self.locally_typical_tau is not None and self.locally_typical_tau <= 0:
            raise ValueError(f"locally_typical_tau must be positive, got {self.locally_typical_tau}")


class Sampler:
    """Base sampler class with various sampling strategies."""
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: SamplingConfig,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[torch.Tensor]]:
        """
        Generate text using specified sampling configuration.
        
        Returns:
            Tuple of (generated_text, list of logits for each step)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        config.validate()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = input_ids.clone()
        past_key_values = None
        all_logits = []
        
        # Track generated n-grams for repetition control
        generated_ngrams = set()
        if config.no_repeat_ngram_size is not None:
            generated_ngrams = self._build_ngram_set(
                generated[0].tolist(), 
                config.no_repeat_ngram_size
            )
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # Only pass the last token when we have cached KV
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            outputs = self.model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs["logits"][0, -1, :]  # (vocab_size,)
            past_key_values = outputs["past_key_values"]
            all_logits.append(logits.cpu())
            
            # Apply sampling strategy
            next_token_logits = logits.clone()
            
            # Apply repetition penalty if specified
            if config.repetition_penalty is not None:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, 
                    generated[0], 
                    config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply no-repeat n-gram constraint
            if config.no_repeat_ngram_size is not None:
                next_token_logits = self._apply_no_repeat_ngrams(
                    next_token_logits,
                    generated[0],
                    generated_ngrams,
                    config.no_repeat_ngram_size
                )
            
            # Sample next token
            next_token = self._sample_token(next_token_logits, config)
            
            # Update generated n-grams
            if config.no_repeat_ngram_size is not None:
                new_sequence = torch.cat([generated[0], next_token])
                self._update_ngram_set(
                    generated_ngrams,
                    new_sequence.tolist(),
                    config.no_repeat_ngram_size
                )
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.special_tokens.get("<EOS>", -1):
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        
        return generated_text, all_logits
    
    def _sample_token(
        self, 
        logits: torch.Tensor, 
        config: SamplingConfig
    ) -> torch.Tensor:
        """Sample next token based on configuration."""
        
        # Apply top-k filtering
        if config.top_k is not None:
            logits = self._top_k_filtering(logits, config.top_k)
        
        # Apply top-p (nucleus) filtering
        if config.top_p is not None:
            logits = self._top_p_filtering(logits, config.top_p)
        
        # Apply locally typical sampling
        if config.locally_typical_tau is not None:
            logits = self._locally_typical_filtering(
                logits, 
                config.locally_typical_tau,
                config.temperature
            )
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token[0]
    
    @staticmethod
    def _top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top k tokens with highest probabilities."""
        if k <= 0:
            return logits
        
        # Get top k tokens
        top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))
        threshold = top_k_values[-1]
        
        # Mask out tokens below threshold
        logits[logits < threshold] = -float('Inf')
        return logits
    
    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling: keep smallest set of tokens with cumulative probability >= p."""
        if p <= 0 or p > 1:
            return logits
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # Get cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find tokens to remove
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create mask
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        
        return logits
    
    @staticmethod
    def _locally_typical_filtering(
        logits: torch.Tensor, 
        tau: float,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Locally typical sampling.
        Filters tokens based on how close their probability is to the local entropy.
        """
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy of the distribution
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs)
        
        # Compute absolute difference between -log prob and entropy
        abs_diff = torch.abs(-log_probs - entropy)
        
        # Keep tokens where difference <= tau
        mask = abs_diff <= tau
        
        # If no tokens satisfy condition, fallback to temperature sampling
        if not mask.any():
            return logits / temperature
        
        return torch.where(mask, logits, torch.full_like(logits, -float('Inf')))
    
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        Apply repetition penalty to previously generated tokens.
        Penalty > 1 discourages repetition, penalty < 1 encourages repetition.
        """
        if penalty == 1.0:
            return logits
        
        # Get unique tokens in generated sequence
        unique_tokens = torch.unique(generated_ids)
        
        # Apply penalty
        for token in unique_tokens:
            if logits[token] < 0:
                logits[token] *= penalty
            else:
                logits[token] /= penalty
        
        return logits
    
    def _build_ngram_set(
        self, 
        token_ids: List[int], 
        ngram_size: int
    ) -> set:
        """Build set of n-grams from token sequence."""
        ngrams = set()
        if len(token_ids) < ngram_size:
            return ngrams
        
        for i in range(len(token_ids) - ngram_size + 1):
            ngram = tuple(token_ids[i:i + ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def _update_ngram_set(
        self,
        ngram_set: set,
        token_ids: List[int],
        ngram_size: int
    ):
        """Update n-gram set with new token."""
        if len(token_ids) < ngram_size:
            return
        
        # Add the most recent n-gram
        ngram = tuple(token_ids[-(ngram_size):])
        ngram_set.add(ngram)
    
    def _apply_no_repeat_ngrams(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        ngram_set: set,
        ngram_size: int
    ) -> torch.Tensor:
        """Prevent repeating n-grams by masking out tokens that would create repetition."""
        if len(generated_ids) < ngram_size - 1:
            return logits
        
        # Get the last (ngram_size - 1) tokens
        context = generated_ids[-(ngram_size - 1):].tolist()
        
        # For each possible next token, check if it would create a repeated n-gram
        for token_id in range(logits.size(-1)):
            candidate_ngram = tuple(context + [token_id])
            if candidate_ngram in ngram_set:
                logits[token_id] = -float('Inf')
        
        return logits


class GreedySampler(Sampler):
    """Greedy sampling (always pick highest probability token)."""
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[SamplingConfig] = None,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[torch.Tensor]]:
        """
        Greedy generation (argmax at each step).
        Note: config is ignored for greedy sampling.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = input_ids.clone()
        past_key_values = None
        all_logits = []
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            outputs = self.model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs["logits"][0, -1, :]
            past_key_values = outputs["past_key_values"]
            all_logits.append(logits.cpu())
            
            # Greedy selection
            next_token = torch.argmax(logits, dim=-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.special_tokens.get("<EOS>", -1):
                break
        
        generated_text = self.tokenizer.decode(generated[0].tolist())
        
        return generated_text, all_logits


class RandomSampler(Sampler):
    """Pure random sampling from the full distribution with temperature."""
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: SamplingConfig,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[torch.Tensor]]:
        """
        Random sampling from the full distribution with temperature.
        Note: Only uses temperature from config, other parameters are ignored.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create a simple config with just temperature
        simple_config = SamplingConfig(temperature=config.temperature)
        
        # Use base class generation with only temperature
        return super().generate(prompt, simple_config, max_new_tokens, seed)
