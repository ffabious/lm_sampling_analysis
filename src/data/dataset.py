import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class LMDataset(Dataset):

    def __init__(self, split_dir):
        self.split_dir = Path(split_dir)
        self.shard_paths = sorted(self.split_dir.glob('shard_*.npy'))
        if not self.shard_paths:
            raise ValueError(f"No shard files found in {self.split_dir}")
        
        self.shard_sizes = []
        self.shard_offsets = [0]
        for shard_path in self.shard_paths:
            shard = np.load(shard_path, mmap_mode='r')
            n, _ = shard.shape
            self.shard_sizes.append(n)
            self.shard_offsets.append(self.shard_offsets[-1] + n)
        self.total_size = self.shard_offsets[-1]

        self._cache_shard_idx = None
        self._cache_arr = None

    def __len__(self):
        return self.total_size
    
    def _locate(self, idx):
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.total_size}")
        shard_idx = int(np.searchsorted(self.shard_offsets, idx, side='right') - 1)
        local_idx = int(idx - self.shard_offsets[shard_idx])
        return shard_idx, local_idx
    
    def _get_shard(self, shard_idx):
        if self._cache_shard_idx == shard_idx and self._cache_arr is not None:
            return self._cache_arr
        
        shard_path = self.shard_paths[shard_idx]
        arr = np.load(shard_path, mmap_mode='r')
        self._cache_shard_idx = shard_idx
        self._cache_arr = arr
        return arr
    
    def __getitem__(self, idx):
        shard_idx, local_idx = self._locate(idx)
        shard = self._get_shard(shard_idx)

        tokens = shard[local_idx].copy()
        tokens_tensor = torch.from_numpy(tokens).long()

        input_ids = tokens_tensor[:-1].contiguous()
        labels = tokens_tensor[1:].contiguous()
        return {
            'input_ids': input_ids,
            'labels': labels
        }