from src.data.tokenizer import LMTokenizer
from datasets import load_dataset
import numpy as np

if __name__ == "__main__":
    tokenizer = LMTokenizer()
    dataset = load_dataset("roneneldan/TinyStories", split='validation')
    corpus_sents = dataset['text']
    tokenizer.train(corpus_sents)
    tokenizer.to_json("checkpoints/tokenizer.json")