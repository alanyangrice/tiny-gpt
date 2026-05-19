"""
Supervised fine-tuning (instruction tuning) for pretrained GPT checkpoints.

Loads a Hugging Face instruction dataset, formats each example as:

    User: ...
    Assistant: ...

and trains only on assistant response tokens by masking prompt labels with -100.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import tiktoken
import torch
from torch import Tensor
from tqdm import tqdm

from config import GPTConfig
from model import GPT
from train import _get_param_groups, get_lr

ENC = tiktoken.get_encoding("gpt2")

SFT_PRESETS: dict[str, dict[str, str | None]] = {
    "alpaca": {
        "dataset": "tatsu-lab/alpaca",
        "split": "train",
        "format": "fields",
        "instruction_field": "instruction",
        "input_field": "input",
        "response_field": "output",
    },
    "dolly": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "format": "fields",
        "instruction_field": "instruction",
        "input_field": "context",
        "response_field": "response",
    },
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "format": "messages",
        "instruction_field": None,
        "input_field": None,
        "response_field": None,
    },
    "openassistant": {
        "dataset": "OpenAssistant/oasst1",
        "split": "train",
        "format": "oasst",
        "instruction_field": None,
        "input_field": None,
        "response_field": None,
    },
}


def format_chat_prompt(prompt: str, system: str | None = None) -> str:
    """Return the chat-style prompt used for SFT and chat inference."""
    prompt = prompt.strip()
    if system:
        return f"System: {system.strip()}\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]


class SFTDataset:
    """Tokenized in-memory instruction dataset with assistant-only labels."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        block_size: int,
        instruction_field: str,
        input_field: str | None,
        response_field: str,
        system: str | None = None,
    ):
        self.block_size = block_size
        self.examples = [
            self._tokenize_example(
                row=row,
                instruction_field=instruction_field,
                input_field=input_field,
                response_field=response_field,
                system=system,
            )
            for row in examples
        ]
        self.examples = [ex for ex in self.examples if ex is not None]
        if not self.examples:
            raise ValueError("No usable SFT examples found after tokenization.")

    def _tokenize_example(
        self,
        row: dict[str, Any],
        instruction_field: str,
        input_field: str | None,
        response_field: str,
        system: str | None,
    ) -> SFTExample | None:
        instruction = str(row.get(instruction_field, "")).strip()
        response = str(row.get(response_field, "")).strip()
        if not instruction or not response:
            return None

        extra_input = ""
        if input_field:
            extra_input = str(row.get(input_field, "")).strip()
        user_text = f"{instruction}\n\n{extra_input}" if extra_input else instruction

        prompt = format_chat_prompt(user_text, system=system)
        prompt_tokens = ENC.encode_ordinary(prompt)
        response_tokens = ENC.encode_ordinary(f" {response}") + [ENC.eot_token]
        full_tokens = prompt_tokens + response_tokens
        if len(full_tokens) < 2:
            return None

        full_tokens = full_tokens[: self.block_size + 1]
        input_ids = full_tokens[:-1]
        labels = full_tokens[1:]

        # labels[i] is the token after input_ids[i]. The first assistant token is
        # full_tokens[len(prompt_tokens)], so its prediction lives at labels[prompt_len - 1].
        first_response_label_idx = max(len(prompt_tokens) - 1, 0)
        labels = [
            token if i >= first_response_label_idx else -100
            for i, token in enumerate(labels)
        ]

        if all(label == -100 for label in labels):
            return None
        return SFTExample(input_ids=input_ids, labels=labels)

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, indices: list[int], device: torch.device) -> tuple[Tensor, Tensor]:
        batch = [self.examples[i] for i in indices]
        max_len = max(len(ex.input_ids) for ex in batch)
        x = torch.full((len(batch), max_len), ENC.eot_token, dtype=torch.long)
        y = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for row_idx, ex in enumerate(batch):
            seq_len = len(ex.input_ids)
            x[row_idx, :seq_len] = torch.tensor(ex.input_ids, dtype=torch.long)
            y[row_idx, :seq_len] = torch.tensor(ex.labels, dtype=torch.long)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def load_hf_examples(
    dataset_name: str,
    split: str,
    dataset_config: str | None,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, name=dataset_config, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return [dict(row) for row in ds]


def normalize_examples(
    rows: list[dict[str, Any]],
    data_format: str,
    instruction_field: str | None,
    input_field: str | None,
    response_field: str | None,
) -> list[dict[str, str]]:
    """Convert supported HF dataset schemas into instruction/input/output rows."""
    if data_format == "fields":
        if not instruction_field or not response_field:
            raise ValueError("Field-format SFT requires --instruction_field and --response_field.")
        return [
            {
                "instruction": str(row.get(instruction_field, "")),
                "input": str(row.get(input_field, "")) if input_field else "",
                "output": str(row.get(response_field, "")),
            }
            for row in rows
        ]

    if data_format == "messages":
        examples = []
        for row in rows:
            messages = row.get("messages", [])
            if not isinstance(messages, list):
                continue
            for i in range(len(messages) - 1):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                if not isinstance(user_msg, dict) or not isinstance(assistant_msg, dict):
                    continue
                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                    examples.append(
                        {
                            "instruction": str(user_msg.get("content", "")),
                            "input": "",
                            "output": str(assistant_msg.get("content", "")),
                        }
                    )
        return examples

    if data_format == "oasst":
        by_id = {row.get("message_id"): row for row in rows}
        examples = []
        for row in rows:
            if row.get("role") != "assistant":
                continue
            parent = by_id.get(row.get("parent_id"))
            if not parent or parent.get("role") not in ("prompter", "user"):
                continue
            examples.append(
                {
                    "instruction": str(parent.get("text", "")),
                    "input": "",
                    "output": str(row.get("text", "")),
                }
            )
        return examples

    raise ValueError(f"Unsupported --format '{data_format}'. Choose fields, messages, or oasst.")


def apply_preset(args) -> None:
    if args.preset == "custom":
        return
    preset = SFT_PRESETS[args.preset]
    args.dataset = preset["dataset"]
    args.split = preset["split"]
    args.format = preset["format"]
    args.instruction_field = preset["instruction_field"]
    args.input_field = preset["input_field"]
    args.response_field = preset["response_field"]


def split_examples(examples: list[dict[str, Any]], val_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_val = max(1, int(len(examples) * val_frac))
    if n_val >= len(examples):
        raise ValueError("Need at least two examples to create a train/validation split.")
    return examples[:-n_val], examples[-n_val:]


def estimate_loss(model: GPT, dataset: SFTDataset, batch_size: int, eval_steps: int, device: torch.device, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        idx = torch.randint(0, len(dataset), (batch_size,)).tolist()
        x, y = dataset.get_batch(idx, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, config: GPTConfig, step: int, val_loss: float, out_dir: str, name: str = "best.pt") -> None:
    os.makedirs(out_dir, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
            "stage": "sft",
        },
        os.path.join(out_dir, name),
    )


def train_sft(args) -> None:
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config: GPTConfig = checkpoint["config"]
    config.use_gradient_checkpointing = args.gradient_checkpointing

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)
    print(f"Using dtype: {dtype}")

    apply_preset(args)
    print(f"SFT preset: {args.preset}")
    print(f"Dataset: {args.dataset} [{args.split}] ({args.format})")

    raw_examples = load_hf_examples(args.dataset, args.split, args.dataset_config, args.max_examples)
    raw_examples = normalize_examples(
        raw_examples,
        args.format,
        args.instruction_field,
        args.input_field,
        args.response_field,
    )
    train_rows, val_rows = split_examples(raw_examples, args.val_frac)
    train_ds = SFTDataset(
        train_rows,
        config.block_size,
        "instruction",
        "input",
        "output",
        system=args.system,
    )
    val_ds = SFTDataset(
        val_rows,
        config.block_size,
        "instruction",
        "input",
        "output",
        system=args.system,
    )
    print(f"SFT examples: train={len(train_ds):,}, val={len(val_ds):,}")

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(
        _get_param_groups(model, args.weight_decay),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8", newline="") as metrics:
        metrics.write("step,train_loss,train_ppl,val_loss,val_ppl,lr,elapsed_sec,vram_gb\n")

        best_val_loss = float("inf")
        train_iter = cycle(torch.randperm(len(train_ds)).tolist())
        t0 = time.time()
        pbar = tqdm(range(args.max_steps), desc="SFT", unit="step", dynamic_ncols=True)

        for step in pbar:
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(args.grad_accum_steps):
                indices = [next(train_iter) for _ in range(args.batch_size)]
                x, y = train_ds.get_batch(indices, device)
                with ctx:
                    _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            do_eval = step > 0 and step % args.eval_interval == 0
            val_loss = None
            if do_eval:
                val_loss = estimate_loss(model, val_ds, args.batch_size, args.eval_steps, device, ctx)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                tqdm.write(f"step {step:5d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.1f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, config, step, val_loss, args.out_dir)
                    tqdm.write(f"  -> saved best SFT checkpoint (val_loss={val_loss:.4f})")

            log_interval = 10 if step < 100 else 50
            if step % log_interval == 0 or do_eval:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                train_ppl = math.exp(accum_loss) if accum_loss < 20 else float("inf")
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0
                val_loss_str = "" if val_loss is None else f"{val_loss:.4f}"
                val_ppl_str = "" if val_loss is None else f"{math.exp(val_loss):.1f}"
                tqdm.write(
                    f"step {step:5d} | loss {accum_loss:.4f} | ppl {train_ppl:8.1f} "
                    f"| lr {lr:.2e} | vram {mem_gb:.1f}GB"
                )
                metrics.write(
                    f"{step},{accum_loss:.4f},{train_ppl:.1f},{val_loss_str},{val_ppl_str},"
                    f"{lr},{elapsed:.1f},{mem_gb:.2f}\n"
                )
                metrics.flush()

            pbar.set_postfix_str(f"{step + 1:,}/{args.max_steps:,}")

        final_loss = estimate_loss(model, val_ds, args.batch_size, args.eval_steps, device, ctx)
        save_checkpoint(model, optimizer, config, args.max_steps, final_loss, args.out_dir, name="final.pt")
        print(f"SFT complete. Final val loss: {final_loss:.4f} | best val loss: {best_val_loss:.4f}")
        print(f"Metrics saved: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-tune a pretrained GPT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preset", type=str, default="alpaca", choices=[*SFT_PRESETS.keys(), "custom"])
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str, default="sft_checkpoints")
    parser.add_argument("--format", type=str, default="fields", choices=["fields", "messages", "oasst"])
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--input_field", type=str, default="input")
    parser.add_argument("--response_field", type=str, default="output")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train_sft(args)


if __name__ == "__main__":
    main()
