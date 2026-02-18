import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import LMDataset

def create_dataloader(split_dir, batch_size=32, train_mode=True, num_workers=2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if train_mode:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    dataset = LMDataset(split_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader

def build_loaders(batch_size=32, num_workers=2, seed=42):
    train_loader = create_dataloader("data/processed/train", batch_size, True, num_workers, seed)
    val_loader = create_dataloader("data/processed/validation", batch_size, False, num_workers, seed)
    test_loader = create_dataloader("data/processed/test", batch_size, False, num_workers, seed)
    return train_loader, val_loader, test_loader