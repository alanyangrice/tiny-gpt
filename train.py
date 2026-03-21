"""
Training script for GPT with Block Attention Residuals.

Supports BF16 mixed precision, fused AdamW, cosine LR with warmup,
gradient checkpointing, torch.compile, gradient accumulation,
local text files, and HuggingFace streaming datasets.

Usage:
    # Local file
    python train.py --data data/shakespeare.txt --preset small

    # Stream from HuggingFace (OpenWebText + FineWeb-Edu + SlimPajama)
    python train.py --data hf --preset small --max_steps 50000
"""

from __future__ import annotations

import argparse
import math
import os
import time

import tiktoken
import torch
from torch import Tensor
from tqdm import tqdm

from config import GPTConfig, TrainConfig
from data import TextDataset, StreamingMixDataset, DATASET_PRESETS
from data.metrics import MetricsLogger
from model import GPT

ENC = tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# CUDA stream prefetcher
# ---------------------------------------------------------------------------

class BatchPrefetcher:
    """Prefetches the next batch on a separate CUDA stream so data transfer
    overlaps with the current training step's compute."""

    def __init__(self, dataset, batch_size: int, device: torch.device):
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
def estimate_loss(model: GPT, dataset: TextDataset, batch_size: int, eval_steps: int, device: torch.device, ctx) -> tuple[float, float]:
    """Returns (avg_loss, perplexity)."""
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = dataset.get_batch(batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


@torch.no_grad()
def generate_sample(model: GPT, device: torch.device, ctx, prompt: str = "The", max_tokens: int = 120) -> str:
    """Generate a short text sample for qualitative evaluation during training."""
    model.eval()
    tokens = ENC.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with ctx:
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, 40)
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            if next_token.item() == ENC.eot_token:
                break

    model.train()
    return ENC.decode(idx[0].tolist())


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
    if cfg.dropout > 0:
        config.dropout = cfg.dropout
        print(f"Dropout: {cfg.dropout}")

    # ---- auto batch size ----
    batch_size = cfg.batch_size
    if batch_size is None:
        if device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            vram_gb = 0

        if vram_gb >= 30:
            batch_table = {"small": 16, "medium": 8, "large": 4, "xl": 2}
        elif vram_gb >= 22:
            batch_table = {"small": 12, "medium": 6, "large": 2, "xl": 1}
        else:
            batch_table = {"small": 6, "medium": 3, "large": 1, "xl": 1}

        batch_size = batch_table[cfg.preset]
        if device.type not in ("cuda",):
            batch_size = min(batch_size, 4)
    print(f"Batch size: {batch_size}, grad accum steps: {cfg.grad_accum_steps}")
    print(f"Effective batch size: {batch_size * cfg.grad_accum_steps}")

    # ---- data ----
    streaming = (cfg.data == "hf")
    if streaming:
        configs = [DATASET_PRESETS[k] for k in DATASET_PRESETS]
        train_ds, val_ds = StreamingMixDataset.create(configs, config.block_size, device)
        train_token_count = None
    else:
        train_ds = TextDataset(cfg.data, config.block_size, split="train").pin(device)
        val_ds = TextDataset(cfg.data, config.block_size, split="val").pin(device)
        train_token_count = len(train_ds.data)
        print(f"Train tokens: {train_token_count:,}, Val tokens: {len(val_ds.data):,}")

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

    # ---- output dir + metrics logger ----
    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics = MetricsLogger(cfg.out_dir)
    print(f"Metrics CSV: {metrics.path}")

    # ---- prefetcher ----
    prefetcher = BatchPrefetcher(train_ds, batch_size, device)
    prefetcher.prefetch()

    # ---- training loop ----
    model.train()
    best_val_loss = float("inf")
    tokens_per_step = batch_size * cfg.grad_accum_steps * config.block_size
    total_tokens_seen = 0
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

        total_tokens_seen += tokens_per_step
        train_ppl = math.exp(accum_loss) if accum_loss < 20 else float("inf")
        epochs = total_tokens_seen / train_token_count if train_token_count else 0

        # ---- logging ----
        log_interval = 10 if step < 100 else 50
        if step % log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            dt = now - t_last_log
            steps_since = log_interval if step > 0 else 1
            tok_per_sec = tokens_per_step * steps_since / dt if dt > 0 else 0
            elapsed = now - t0

            mem_gb = 0.0
            mem_str = ""
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                mem_str = f" | vram {mem_gb:.1f}GB"

            epoch_str = f" | epoch {epochs:.2f}" if train_token_count else ""

            t_last_log = now
            tqdm.write(
                f"step {step:5d} | loss {accum_loss:.4f} | ppl {train_ppl:8.1f} "
                f"| lr {lr:.2e}{epoch_str} "
                f"| {tok_per_sec:,.0f} tok/s{mem_str}"
            )

            metrics.log(
                step=step, train_loss=round(accum_loss, 4),
                train_ppl=round(train_ppl, 1), lr=lr,
                epoch=round(epochs, 3) if train_token_count else "",
                tok_per_sec=round(tok_per_sec), vram_gb=round(mem_gb, 2),
                elapsed_sec=round(elapsed, 1),
            )

        # ---- eval + sample generation ----
        if step > 0 and step % cfg.eval_interval == 0:
            val_loss, val_ppl = estimate_loss(model, val_ds, batch_size, cfg.eval_steps, device, ctx)
            tqdm.write(f"step {step:5d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.1f}")

            sample = generate_sample(model, device, ctx)
            tqdm.write(f"  sample: {sample[:200]}{'...' if len(sample) > 200 else ''}")

            metrics.log(
                step=step, val_loss=round(val_loss, 4), val_ppl=round(val_ppl, 1),
                elapsed_sec=round(time.time() - t0, 1),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, optimizer, config, step, val_loss, cfg.out_dir)
                tqdm.write(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")

    # ---- final save + summary ----
    val_loss, val_ppl = estimate_loss(model, val_ds, batch_size, cfg.eval_steps, device, ctx)
    _save_checkpoint(model, optimizer, config, cfg.max_steps, val_loss, cfg.out_dir, name="final.pt")

    metrics.log(step=cfg.max_steps, val_loss=round(val_loss, 4), val_ppl=round(val_ppl, 1),
                elapsed_sec=round(time.time() - t0, 1))
    metrics.close()

    if streaming:
        train_ds.stop()

    print(f"\n{'='*60}")
    print(f"Training complete after {cfg.max_steps} steps ({total_tokens_seen/1e6:.1f}M tokens)")
    print(f"  Final val loss: {val_loss:.4f} | val ppl: {val_ppl:.1f}")
    print(f"  Best  val loss: {best_val_loss:.4f}")
    if train_token_count:
        print(f"  Data epochs:    {epochs:.2f}")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  Peak GPU memory: {peak_gb:.2f} GB")
    print(f"  Metrics saved:  {metrics.path}")
    print(f"{'='*60}")

    sample = generate_sample(model, device, ctx)
    print(f"\nFinal sample:\n{sample}")


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
    parser.add_argument("--data", type=str, required=True,
                        help="Path to text file, or 'hf' to stream from HuggingFace")
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
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (0.1 recommended for small datasets)")
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
        dropout=args.dropout,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        compile=args.compile,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
