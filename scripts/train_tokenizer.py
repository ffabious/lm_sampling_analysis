import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import LMTokenizer
from datasets import load_dataset

if __name__ == "__main__":
    tokenizer = LMTokenizer()
    dataset = load_dataset("roneneldan/TinyStories", split='validation')
    corpus_sents = dataset['text']
    tokenizer.train(corpus_sents)
    tokenizer.to_json("checkpoints/tokenizer.json")
