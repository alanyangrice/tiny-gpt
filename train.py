"""
Training script for GPT with Block Attention Residuals.

Supports BF16 mixed precision, fused AdamW, cosine LR with warmup,
gradient checkpointing, torch.compile, and gradient accumulation.

Usage:
    python train.py --data data.txt --preset small
    python train.py --data data.txt --preset medium --compile
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import tiktoken
import torch
from torch import Tensor
from tqdm import tqdm

from config import GPTConfig, TrainConfig
from model import GPT


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TextDataset:
    """Loads a text file, encodes with tiktoken, serves random chunks."""

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


# ---------------------------------------------------------------------------
# CUDA stream prefetcher
# ---------------------------------------------------------------------------

class BatchPrefetcher:
    """Prefetches the next batch on a separate CUDA stream so data transfer
    overlaps with the current training step's compute."""

    def __init__(self, dataset: TextDataset, batch_size: int, device: torch.device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._next_x: Tensor | None = None
        self._next_y: Tensor | None = None

    def prefetch(self) -> None:
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._next_x, self._next_y = self.dataset.get_batch(self.batch_size, self.device)
        else:
            self._next_x, self._next_y = self.dataset.get_batch(self.batch_size, self.device)

    def next(self) -> tuple[Tensor, Tensor]:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self._next_x, self._next_y
        self.prefetch()
        return x, y


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model: GPT, dataset: TextDataset, batch_size: int, eval_steps: int, device: torch.device, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = dataset.get_batch(batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    """Run the full training loop with the given configuration."""

    # ---- device ----
    if cfg.device:
        device = torch.device(cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU: {props.name} ({vram_gb:.1f} GB)")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)
    print(f"Using dtype: {dtype}")

    # ---- config ----
    config_fn = {"small": GPTConfig.small, "medium": GPTConfig.medium, "large": GPTConfig.large, "xl": GPTConfig.xl}
    config = config_fn[cfg.preset]()

    # ---- auto batch size ----
    # AttnRes blocks accumulate and stack all block representations through
    # depth, so activation memory is much higher than a vanilla transformer
    # of the same parameter count.  Batch sizes below are conservative to
    # avoid OOM; increase manually if your GPU has headroom.
    batch_size = cfg.batch_size
    if batch_size is None:
        if device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            vram_gb = 0

        if vram_gb >= 30:         # RTX 5090 / A100 class
            batch_table = {"small": 16, "medium": 8, "large": 4, "xl": 2}
        elif vram_gb >= 22:       # RTX 4090 / 3090 class
            batch_table = {"small": 12, "medium": 6, "large": 2, "xl": 1}
        else:
            batch_table = {"small": 6, "medium": 3, "large": 1, "xl": 1}

        batch_size = batch_table[cfg.preset]
        if device.type not in ("cuda",):
            batch_size = min(batch_size, 4)
    print(f"Batch size: {batch_size}, grad accum steps: {cfg.grad_accum_steps}")
    print(f"Effective batch size: {batch_size * cfg.grad_accum_steps}")

    # ---- data ----
    train_ds = TextDataset(cfg.data, config.block_size, split="train").pin(device)
    val_ds = TextDataset(cfg.data, config.block_size, split="val").pin(device)
    print(f"Train tokens: {len(train_ds.data):,}, Val tokens: {len(val_ds.data):,}")

    # ---- model ----
    model = GPT(config).to(device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if cfg.compile and device.type == "cuda":
        print("Compiling model with torch.compile (reduce-overhead) ...")
        model = torch.compile(model, mode="reduce-overhead")

    # ---- optimizer ----
    param_groups = _get_param_groups(model, cfg.weight_decay)
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.max_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    # ---- output dir ----
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ---- prefetcher ----
    prefetcher = BatchPrefetcher(train_ds, batch_size, device)
    prefetcher.prefetch()

    # ---- training loop ----
    model.train()
    best_val_loss = float("inf")
    tokens_per_step = batch_size * cfg.grad_accum_steps * config.block_size
    t0 = time.time()
    t_last_log = t0

    for step in tqdm(range(cfg.max_steps), desc="Training"):
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y = prefetcher.next()
            with ctx:
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ---- logging ----
        log_interval = 10 if step < 100 else 50
        if step % log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            dt = now - t_last_log
            steps_since = log_interval if step > 0 else 1
            tok_per_sec = tokens_per_step * steps_since / dt if dt > 0 else 0

            mem_str = ""
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                mem_str = f" | vram {mem_gb:.1f}GB"

            t_last_log = now
            tqdm.write(
                f"step {step:5d} | loss {accum_loss:.4f} | lr {lr:.2e} "
                f"| {now - t0:.1f}s | {tok_per_sec:,.0f} tok/s{mem_str}"
            )

        # ---- eval ----
        if step > 0 and step % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_ds, batch_size, cfg.eval_steps, device, ctx)
            tqdm.write(f"step {step:5d} | val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, optimizer, config, step, val_loss, cfg.out_dir)
                tqdm.write(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")

    # final save
    val_loss = estimate_loss(model, val_ds, batch_size, cfg.eval_steps, device, ctx)
    _save_checkpoint(model, optimizer, config, cfg.max_steps, val_loss, cfg.out_dir, name="final.pt")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak GPU memory: {peak_gb:.2f} GB")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Separate parameters into decayed (2D+ weights) and non-decayed (biases, norms, pseudo-queries)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "pseudo_query" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _save_checkpoint(model, optimizer, config, step, val_loss, out_dir, name="best.pt"):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
        "val_loss": val_loss,
    }, os.path.join(out_dir, name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GPT with Block Attention Residuals")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "medium", "large", "xl"])
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro batch size (auto if not set)")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        data=args.data,
        preset=args.preset,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        compile=args.compile,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
