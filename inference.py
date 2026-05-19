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


def format_chat_prompt(prompt: str, system: str | None = None) -> str:
    """Return the User/Assistant prompt format used by supervised fine-tuning."""
    prompt = prompt.strip()
    if system:
        return f"System: {system.strip()}\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


def load_model(
    checkpoint_path: str,
    device: torch.device,
    compile_model: bool = False,
) -> tuple[GPT, tiktoken.Encoding]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: GPTConfig = checkpoint["config"]
    config.use_gradient_checkpointing = False

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=model_dtype)
    model.eval()

    if compile_model and device.type == "cuda":
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

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
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    stop_on_eot: bool = True,
) -> str:
    tokens = enc.encode_ordinary(prompt)
    prompt_len = len(tokens)
    total_len = prompt_len + max_new_tokens

    buf = torch.zeros(1, total_len, dtype=torch.long, device=device)
    buf[0, :prompt_len] = torch.tensor(tokens, dtype=torch.long, device=device)
    cur_len = prompt_len

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    ctx = torch.amp.autocast(device_type=device.type, dtype=dtype) if device.type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)

    if stream:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    block_size = model.config.block_size
    with torch.no_grad(), ctx:
        for _ in range(max_new_tokens):
            start = max(0, cur_len - block_size)
            idx_cond = buf[:, start:cur_len]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            generated = buf[0, :cur_len]

            if repetition_penalty != 1.0:
                seen_tokens = torch.unique(generated)
                token_logits = logits[0, seen_tokens]
                logits[0, seen_tokens] = torch.where(
                    token_logits < 0,
                    token_logits * repetition_penalty,
                    token_logits / repetition_penalty,
                )

            if no_repeat_ngram_size > 0 and cur_len >= no_repeat_ngram_size - 1:
                banned = _get_banned_ngram_tokens(generated.tolist(), no_repeat_ngram_size)
                if banned:
                    logits[0, banned] = float("-inf")

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
            if stop_on_eot and next_token.item() == enc.eot_token:
                break
            buf[0, cur_len] = next_token.squeeze()
            cur_len += 1

            if stream:
                tok_str = enc.decode([next_token.item()])
                sys.stdout.write(tok_str)
                sys.stdout.flush()

    text = enc.decode(buf[0, :cur_len].tolist())

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return text


def _get_banned_ngram_tokens(tokens: list[int], ngram_size: int) -> list[int]:
    """Return tokens that would repeat an existing n-gram if sampled next."""
    if ngram_size <= 0 or len(tokens) < ngram_size - 1:
        return []

    prefix_len = ngram_size - 1
    current_prefix = tuple(tokens[-prefix_len:]) if prefix_len > 0 else tuple()
    banned = []
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tokens[i : i + ngram_size]
        if tuple(ngram[:-1]) == current_prefix:
            banned.append(ngram[-1])
    return banned


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained GPT + AttnRes model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--no_stop_on_eot", action="store_true", help="Do not stop when GPT-2 EOT is sampled")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chat", action="store_true", help="Wrap prompt as 'User: ...\\nAssistant:'")
    parser.add_argument("--system", type=str, default=None, help="Optional system message for --chat")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, enc = load_model(args.checkpoint, device, compile_model=args.compile)
    prompt = format_chat_prompt(args.prompt, args.system) if args.chat else args.prompt

    print(f"--- Generating (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}) ---")
    text = generate(
        model, enc, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        stream=not args.no_stream,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        stop_on_eot=not args.no_stop_on_eot,
    )

    if args.no_stream:
        print(text)


if __name__ == "__main__":
    main()
