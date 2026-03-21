"""Local text file dataset -- loads entirely into memory, serves random chunks."""

from __future__ import annotations

import tiktoken
import torch
from torch import Tensor


class TextDataset:
    """Loads a text file, encodes with tiktoken GPT-2 BPE, serves random chunks."""

    def __init__(self, path: str, block_size: int, split: str = "train", val_frac: float = 0.05):
        enc = tiktoken.get_encoding("gpt2")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = enc.encode_ordinary(text)
        data = torch.tensor(tokens, dtype=torch.long)

        n_val = max(int(len(data) * val_frac), block_size + 1)
        if split == "train":
            self.data = data[:-n_val] if n_val < len(data) else data
        else:
            self.data = data[-n_val:]

        assert len(self.data) > block_size, (
            f"Dataset split '{split}' has {len(self.data)} tokens but "
            f"block_size is {block_size}. Provide more data."
        )
        self.block_size = block_size

    def pin(self, device: torch.device) -> "TextDataset":
        """Move data to GPU if it fits comfortably (< 512 MB), otherwise pin to host memory."""
        size_mb = self.data.nbytes / (1024 * 1024)
        if device.type == "cuda" and size_mb < 512:
            self.data = self.data.to(device)
        elif device.type == "cuda":
            self.data = self.data.pin_memory()
        self._offsets = torch.arange(self.block_size, device=self.data.device)
        return self

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        ix = torch.randint(0, len(self), (batch_size,), device=self.data.device)
        indices = ix.unsqueeze(1) + self._offsets.unsqueeze(0)
        x = self.data[indices]
        y = self.data[indices + 1]
        if self.data.device == device:
            return x, y
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
