import re
from typing import Dict, List, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import bleuscore


def tokenize_for_metrics(text: str) -> List[str]:
    """Stable word/punctuation tokenizer used by non-model text metrics."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class PerplexityEvaluator:
    """Compute perplexity using a larger reference model."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "cpu",
        model=None,
        tokenizer=None,
        batch_size: int = 8,
    ):
        self.device = device
        self.batch_size = batch_size
        self.use_local_model = model is not None and tokenizer is not None

        if self.use_local_model:
            self.model = model
            self.tokenizer = tokenizer
            self.model.eval()
            print("Using local GPT model/tokenizer for perplexity evaluation.")
            return

        print(f"Loading reference model '{model_name}' for perplexity evaluation...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print(f"Loading tokenizer for '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("Reference model and tokenizer loaded successfully.")
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def compute_perplexity(self, texts: List[str]) -> float:
        """Compute average perplexity for a list of texts."""
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            return float("inf")

        total_loss = 0.0
        total_tokens = 0

        if self.use_local_model:
            for text in texts:
                token_ids = self.tokenizer.encode(text, add_eos=False)
                if len(token_ids) < 2:
                    continue

                max_seq_len = getattr(self.model.config, "max_seq_len", 256)
                window = max_seq_len + 1

                for start in range(0, len(token_ids) - 1, max_seq_len):
                    segment = token_ids[start:start + window]
                    if len(segment) < 2:
                        continue

                    inputs = torch.tensor([segment[:-1]], dtype=torch.long, device=self.device)
                    labels = torch.tensor([segment[1:]], dtype=torch.long, device=self.device)

                    outputs = self.model(inputs, labels=labels)
                    loss = outputs["loss"]
                    seq_tokens = labels.size(1)

                    total_loss += loss.item() * seq_tokens
                    total_tokens += seq_tokens

            if total_tokens == 0:
                return float("inf")

            avg_loss = total_loss / total_tokens
            return float(np.exp(avg_loss))
        
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Causal LM loss predicts tokens 1..N-1 from earlier context.
            target_tokens = int(torch.clamp(attention_mask.sum(dim=1) - 1, min=0).sum().item())
            if target_tokens == 0:
                continue
            
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item() * target_tokens
            total_tokens += target_tokens
        
        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        return float(np.exp(avg_loss))


class SelfBleuEvaluator:
    """Compute Self-BLEU score to measure diversity using bleuscore."""
    
    def __init__(self, max_ngram: int = 4):
        self.max_ngram = max_ngram
    
    def compute_self_bleu(self, texts: List[str]) -> float:
        """
        Compute Self-BLEU: average BLEU score of each text against all others.
        Lower scores indicate higher diversity.
        
        Uses bleuscore library for BLEU computation.
        """
        if len(texts) < 2:
            return 0.0
        
        scores = []
        tokenized_texts = [tokenize_for_metrics(text) for text in texts]
        
        for i, hypothesis_tokens in enumerate(tokenized_texts):
            references = tokenized_texts[:i] + tokenized_texts[i+1:]
            if not hypothesis_tokens or not references:
                continue
            
            # Compute BLEU for this text against all others using bleuscore
            # bleuscore expects tokenized references and a tokenized prediction
            bleu_result = bleuscore.compute(
                references=references,
                predictions=hypothesis_tokens,
                max_order=self.max_ngram,
                smooth=True,
            )
            
            scores.append(float(bleu_result["bleu"]))
        
        return float(np.mean(scores)) if scores else 0.0


class RepetitionEvaluator:
    """Measure repetition in generated texts."""
    
    @staticmethod
    def compute_repetition_4(texts: List[str]) -> float:
        """Percentage of texts with repeating 4-grams."""
        repetition_count = 0
        
        for text in texts:
            words = tokenize_for_metrics(text)
            if len(words) < 4:
                continue
            
            # Check for repeating 4-grams
            ngrams_seen = set()
            has_repetition = False
            
            for i in range(len(words) - 3):
                ngram = tuple(words[i:i+4])
                if ngram in ngrams_seen:
                    has_repetition = True
                    break
                ngrams_seen.add(ngram)
            
            if has_repetition:
                repetition_count += 1
        
        return repetition_count / len(texts) if texts else 0.0
    
    @staticmethod
    def compute_distinct_2(texts: List[str]) -> float:
        """
        Distinct-2: ratio of unique 2-grams to total 2-grams.
        Measures vocabulary richness.
        """
        all_ngrams = []
        total_ngrams = 0
        
        for text in texts:
            words = tokenize_for_metrics(text)
            if len(words) < 2:
                continue
            
            for i in range(len(words) - 1):
                ngram = tuple(words[i:i+2])
                all_ngrams.append(ngram)
                total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        return unique_ngrams / total_ngrams


class EvaluationPipeline:
    """Run all evaluations for generated texts."""
    
    def __init__(
        self,
        device: str = "cpu",
        model=None,
        tokenizer=None,
        reference_model_name: str = "distilgpt2",
        perplexity_source: str = "reference",
        perplexity_batch_size: int = 8,
    ):
        if perplexity_source not in {"reference", "local", "none"}:
            raise ValueError(
                "perplexity_source must be one of: reference, local, none"
            )

        self.perplexity_source = perplexity_source
        self.perplexity_eval: Optional[PerplexityEvaluator] = None
        if perplexity_source == "reference":
            self.perplexity_eval = PerplexityEvaluator(
                model_name=reference_model_name,
                device=device,
                batch_size=perplexity_batch_size,
            )
        elif perplexity_source == "local":
            if model is None or tokenizer is None:
                raise ValueError("Local perplexity requires model and tokenizer.")
            self.perplexity_eval = PerplexityEvaluator(
                device=device,
                model=model,
                tokenizer=tokenizer,
                batch_size=perplexity_batch_size,
            )

        self.self_bleu_eval = SelfBleuEvaluator()
        self.repetition_eval = RepetitionEvaluator()
    
    def evaluate(
        self, 
        texts: List[str], 
        compute_perplexity: bool = True
    ) -> Dict[str, float]:
        """Run all metrics on the generated texts."""
        results = {}
        print("Starting evaluation of generated texts...")
        if compute_perplexity and self.perplexity_eval is not None:
            results["perplexity"] = self.perplexity_eval.compute_perplexity(texts)
            print("Perplexity evaluation completed.")
        results["self_bleu"] = self.self_bleu_eval.compute_self_bleu(texts)
        print("Self-BLEU evaluation completed.")
        results["repetition_4"] = self.repetition_eval.compute_repetition_4(texts)
        print("Repetition-4 evaluation completed.")
        results["distinct_2"] = self.repetition_eval.compute_distinct_2(texts)
        print("Distinct-2 evaluation completed.")
        
        return results
