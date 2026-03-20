from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling operations."""
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    locally_typical_tau: Optional[float] = None

    def validate(self):
        """Validates that parameters fall within acceptable mathematical bounds."""
        if self.temperature <= 0:
            raise ValueError(
                f"Temperature must be positive, got {self.temperature}")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be positive, got {self.repetition_penalty}")

        if self.locally_typical_tau is not None and self.locally_typical_tau <= 0:
            raise ValueError(
                f"locally_typical_tau must be positive, got {self.locally_typical_tau}")
