import torch
from torch import nn
import random
import argparse
from tqdm.auto import tqdm
import time

from src.data.tokenizer import LMTokenizer
from src.models.gpt import GPTConfig, GPT
from src.data.loader import build_loaders

def get_vocab_size():
    try:
        tokenizer = LMTokenizer.from_json("checkpoints/tokenizer.json")
    except FileNotFoundError:
        print("Tokenizer file not found. Please run the tokenizer training script first.")
        exit(1)
    return len(tokenizer.vocabulary)

def train_model(batch_size=32, lr=1e-4, epochs=10, num_workers=2, device='cpu', seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, _ = build_loaders(batch_size=batch_size, num_workers=num_workers, seed=seed)

    config = GPTConfig(vocab_size=get_vocab_size())
    model = GPT(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        train_loss_sum = 0.0
        train_tokens = 0
        for batch in pbar:
            inputs, targets = batch['input_ids'], batch['labels']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets)
            loss = outputs['loss']

            ignore_index = -100
            valid = targets.ne(ignore_index)
            num_tokens = int(valid.sum().item())
            train_loss_sum += loss.item() * num_tokens
            train_tokens += num_tokens
            avg_train_loss = train_loss_sum / max(train_tokens, 1)
            pbar.set_postfix({"loss": avg_train_loss})
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = train_loss_sum / max(train_tokens, 1)
        print(f"Epoch {epoch + 1} train loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.inference_mode():
            for batch in val_loader:
                inputs, targets = batch["input_ids"], batch["labels"]
                inputs, targets = inputs.to(device), targets.to(device)

                out = model(inputs, labels=targets)
                loss = out["loss"]

                ignore_index = -100
                valid = targets.ne(ignore_index)
                num_tokens = int(valid.sum().item())

                val_loss_sum += loss.item() * num_tokens
                val_tokens += num_tokens

        avg_val_loss = val_loss_sum / max(val_tokens, 1)
        print(f"Validation loss after epoch {epoch + 1}: {avg_val_loss:.4f}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"gpt_model_{timestamp}.pth")
    print(f"Model saved as gpt_model_{timestamp}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    args = parser.parse_args()
    train_model(
        batch_size=args.batch_size,
        lr=args.lr, epochs=args.epochs,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed
    )