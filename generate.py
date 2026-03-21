"""
Text generation script for GPT with Block Attention Residuals.

Loads a checkpoint and generates text autoregressively with configurable
temperature, top-k, and top-p sampling.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
    python generate.py --checkpoint checkpoints/best.pt --prompt "The meaning of life" --top_k 50 --temperature 0.8
"""

from __future__ import annotations

import argparse
import sys

import tiktoken
import torch

from config import GPTConfig
from model import GPT


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, tiktoken.Encoding]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: GPTConfig = checkpoint["config"]
    config.use_gradient_checkpointing = False

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Loaded checkpoint from step {step} (val_loss={val_loss})")
    return model, enc


def generate(
    model: GPT,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device = torch.device("cpu"),
    stream: bool = True,
) -> str:
    tokens = enc.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    if stream:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if stream:
                tok_str = enc.decode([next_token.item()])
                sys.stdout.write(tok_str)
                sys.stdout.flush()

    generated_tokens = idx[0].tolist()
    text = enc.decode(generated_tokens)

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained GPT + AttnRes model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, enc = load_model(args.checkpoint, device)

    print(f"--- Generating (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}) ---")
    text = generate(
        model, enc, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        stream=not args.no_stream,
    )

    if args.no_stream:
        print(text)


if __name__ == "__main__":
    main()
