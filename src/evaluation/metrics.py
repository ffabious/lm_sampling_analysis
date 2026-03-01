import torch
import numpy as np
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import bleuscore

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PerplexityEvaluator:
    """Compute perplexity using a larger reference model."""
    
    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        self.device = device
        print(f"Loading reference model '{model_name}' for perplexity evaluation...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print(f"Loading tokenizer for '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Reference model and tokenizer loaded successfully.")
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def compute_perplexity(self, texts: List[str]) -> float:
        """Compute average perplexity for a list of texts."""
        total_loss = 0.0
        total_tokens = 0
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Perplexity = exp(loss)
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
        
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
        tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
        
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
            words = nltk.word_tokenize(text.lower())
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
            words = nltk.word_tokenize(text.lower())
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
    
    def __init__(self, device: str = "cpu"):
        self.perplexity_eval = PerplexityEvaluator(device=device)
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
        if compute_perplexity:
            results["perplexity"] = self.perplexity_eval.compute_perplexity(texts)
        print("Perplexity evaluation completed.")
        results["self_bleu"] = self.self_bleu_eval.compute_self_bleu(texts)
        print("Self-BLEU evaluation completed.")
        results["repetition_4"] = self.repetition_eval.compute_repetition_4(texts)
        print("Repetition-4 evaluation completed.")
        results["distinct_2"] = self.repetition_eval.compute_distinct_2(texts)
        print("Distinct-2 evaluation completed.")
        
        return results
