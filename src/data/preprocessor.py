import os
import json
from datasets import load_dataset
from datasets.utils.logging import enable_progress_bar
import numpy as np
from tqdm.auto import tqdm


def preprocess(tokenizer, block_size=256, splits=["train", "validation"], shard_size=65536):
    metadata = {
        "dataset": "roneneldan/TinyStories",
        "requested_splits": list(splits),
        "saved_splits": [],
        "block_size": block_size,
        "shard_size": shard_size,
        "dtype": "uint16",
        "num_sequences": {},
        "num_shards": {}
    }

    for split in splits:
        if split == "validation":
            validation_data, test_data = process_validation_split(tokenizer, block_size)

            os.makedirs("data/processed/validation", exist_ok=True)
            for i in tqdm(range(0, len(validation_data), shard_size), desc="Sharding validation"):
                shard = validation_data[i:i + shard_size]
                np.save(f"data/processed/validation/shard_{i // shard_size}.npy", shard)
            metadata["saved_splits"].append("validation")
            metadata["num_sequences"]["validation"] = int(len(validation_data))
            metadata["num_shards"]["validation"] = int((len(validation_data) + shard_size - 1) // shard_size)

            os.makedirs("data/processed/test", exist_ok=True)
            for i in tqdm(range(0, len(test_data), shard_size), desc="Sharding test"):
                shard = test_data[i:i + shard_size]
                np.save(f"data/processed/test/shard_{i // shard_size}.npy", shard)
            metadata["saved_splits"].append("test")
            metadata["num_sequences"]["test"] = int(len(test_data))
            metadata["num_shards"]["test"] = int((len(test_data) + shard_size - 1) // shard_size)
            continue

        tokenized_data = process_split(tokenizer, block_size, split)
        os.makedirs(f"data/processed/{split}", exist_ok=True)
        for i in tqdm(range(0, len(tokenized_data), shard_size), desc=f"Sharding {split}"):
            shard = tokenized_data[i:i + shard_size]
            np.save(f"data/processed/{split}/shard_{i // shard_size}.npy", shard)
        metadata["saved_splits"].append(split)
        metadata["num_sequences"][split] = int(len(tokenized_data))
        metadata["num_shards"][split] = int((len(tokenized_data) + shard_size - 1) // shard_size)

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def process_validation_split(tokenizer, block_size):
    enable_progress_bar()
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    token_stream = []
    for text in dataset['text']:
        token_stream.extend(tokenizer.encode(text, add_eos=True))

    tokenized_data = []
    for i in tqdm(range(0, len(token_stream) - block_size, block_size), desc="Tokenizing validation"):
        block = token_stream[i:i + block_size + 1]
        tokenized_data.append(block)

    tokenized_data = np.array(tokenized_data, dtype=np.uint16)
    split_idx = len(tokenized_data) // 2
    return tokenized_data[:split_idx], tokenized_data[split_idx:]


def process_split(tokenizer, block_size, split):
    enable_progress_bar()
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    token_stream = []
    for text in dataset['text']:
        token_stream.extend(tokenizer.encode(text, add_eos=True))

    tokenized_data = []
    for i in tqdm(range(0, len(token_stream) - block_size, block_size), desc=f"Tokenizing {split}"):
        block = token_stream[i:i + block_size + 1]
        tokenized_data.append(block)

    return np.array(tokenized_data, dtype=np.uint16)
