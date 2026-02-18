from src.data.tokenizer import LMTokenizer
from src.data.preprocessor import preprocess

if __name__ == "__main__":
    print("Loading tokenizer...")
    try:
        tokenizer = LMTokenizer.from_json("checkpoints/tokenizer.json")
    except FileNotFoundError:
        print("Tokenizer file not found. Please run the tokenizer training script first.")
        exit(1)
    print("Tokenizer loaded successfully.")
    print("Preprocessing data...")
    preprocess(tokenizer, splits=['train', 'validation'], block_size=256, shard_size=65536)
    print("Data preprocessing completed.")