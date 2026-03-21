"""
Streaming dataset that mixes multiple HuggingFace datasets with weighted
sampling, tokenizes on-the-fly, and serves batches from a background-filled
token buffer.  Zero disk usage -- everything streams over the network.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import tiktoken
import torch
from torch import Tensor

if TYPE_CHECKING:
    from data.registry import HFDatasetConfig
    from data.text import TextDataset


class StreamingMixDataset:
    """
    Streams text from one or more HuggingFace datasets, tokenizes with
    tiktoken, and serves sequential (x, y) chunks of ``block_size`` tokens.

    A background thread keeps a token ring-buffer filled so ``get_batch``
    never blocks (as long as the network can keep up).
    """

    def __init__(
        self,
        configs: list[HFDatasetConfig],
        block_size: int,
        buffer_tokens: int = 2_000_000,
    ):
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.configs = configs
        self._buf_cap = buffer_tokens
        self._buf = torch.zeros(buffer_tokens, dtype=torch.long)
        self._write = 0
        self._read = 0
        self._lock = threading.Lock()
        self._has_data = threading.Event()
        self._need_data = threading.Event()
        self._need_data.set()
        self._stop = False

        self._stream_iter = self._build_stream()
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Build the interleaved HF stream
    # ------------------------------------------------------------------

    def _build_stream(self):
        from datasets import load_dataset, interleave_datasets

        streams, probs = [], []
        for cfg in self.configs:
            ds = load_dataset(
                cfg.path,
                name=cfg.config_name,
                split=cfg.split,
                streaming=True,
                trust_remote_code=True,
            )
            streams.append(ds)
            probs.append(cfg.weight)

        total = sum(probs)
        probs = [p / total for p in probs]

        if len(streams) == 1:
            mixed = streams[0]
        else:
            mixed = interleave_datasets(streams, probabilities=probs, stopping_strategy="all_exhausted")

        text_field = self.configs[0].text_field
        for doc in mixed:
            text = doc.get(text_field, "")
            if text:
                yield text

    # ------------------------------------------------------------------
    # Background thread: tokenize and fill the ring buffer
    # ------------------------------------------------------------------

    def _fill_loop(self) -> None:
        for text in self._stream_iter:
            if self._stop:
                return
            tokens = self.enc.encode_ordinary(text)
            if not tokens:
                continue
            t = torch.tensor(tokens, dtype=torch.long)
            offset = 0
            while offset < len(t):
                with self._lock:
                    available = self._buf_cap - (self._write - self._read)
                if available <= 0:
                    self._need_data.wait(timeout=0.1)
                    self._need_data.clear()
                    continue
                chunk = t[offset : offset + available]
                n = len(chunk)
                start = self._write % self._buf_cap
                end = start + n
                if end <= self._buf_cap:
                    self._buf[start:end] = chunk
                else:
                    first = self._buf_cap - start
                    self._buf[start:] = chunk[:first]
                    self._buf[: n - first] = chunk[first:]
                with self._lock:
                    self._write += n
                self._has_data.set()
                offset += n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def _available(self) -> int:
        with self._lock:
            return self._write - self._read

    def _wait_for(self, n: int, timeout: float = 60.0) -> None:
        """Block until at least *n* tokens are available in the buffer."""
        import time
        deadline = time.monotonic() + timeout
        while self._available < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Streaming buffer has {self._available} tokens but "
                    f"need {n}. Check your internet connection."
                )
            self._has_data.wait(timeout=min(remaining, 1.0))
            self._has_data.clear()

    def _consume(self, n: int) -> Tensor:
        """Read *n* tokens from the ring buffer (non-wrapping copy out)."""
        start = self._read % self._buf_cap
        end = start + n
        if end <= self._buf_cap:
            out = self._buf[start:end].clone()
        else:
            first = self._buf_cap - start
            out = torch.cat([self._buf[start:].clone(), self._buf[: n - first].clone()])
        with self._lock:
            self._read += n
        self._need_data.set()
        return out

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return (x, y) each of shape [batch_size, block_size]."""
        total = batch_size * (self.block_size + 1)
        self._wait_for(total)
        flat = self._consume(total)
        flat = flat.view(batch_size, self.block_size + 1)
        x = flat[:, :-1]
        y = flat[:, 1:]
        return x.to(device), y.to(device)

    def stop(self) -> None:
        self._stop = True
        self._need_data.set()

    # ------------------------------------------------------------------
    # Factory: create train + val pair
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        configs: list[HFDatasetConfig],
        block_size: int,
        device: torch.device,
        val_tokens: int = 50_000,
    ) -> tuple["StreamingMixDataset", "TextDataset"]:
        """
        Create a streaming training dataset and a small fixed validation
        dataset (extracted from the first tokens of the stream).
        """
        from data.text import TextDataset as _TD

        names = ", ".join(c.name for c in configs)
        weights = ", ".join(f"{c.weight:.0%}" for c in configs)
        print(f"Streaming datasets: {names}")
        print(f"Mixing weights:     {weights}")
        print(f"Buffering {val_tokens:,} validation tokens from stream...")

        stream_ds = cls(configs, block_size)
        stream_ds._wait_for(val_tokens)
        val_data = stream_ds._consume(val_tokens)

        val_ds = object.__new__(_TD)
        val_ds.data = val_data
        val_ds.block_size = block_size
        val_ds.pin(device)

        print(f"Validation tokens: {len(val_ds.data):,}")
        print("Streaming training data (continuous, no epoch boundary)")
        return stream_ds, val_ds
