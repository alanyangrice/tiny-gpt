# TinyGPT

A GPT language model built from scratch in PyTorch, using [Moonshot AI's Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) to replace standard residual connections with learned depth-wise softmax attention.

The model uses a modern LLaMA-style architecture (RoPE, RMSNorm, SwiGLU, Grouped-Query Attention, Flash Attention) and supports streaming pre-training from HuggingFace datasets with zero disk usage.

## Architecture

Standard transformers accumulate layer outputs with a fixed residual: `h = h + layer(h)`. Every layer gets the same uniform sum of everything before it. Attention Residuals replace this with a **learned, input-dependent** weighted sum over depth -- each layer decides how much to draw from each previous block via softmax attention with a per-layer pseudo-query vector.

In practice, we use **Block Attention Residuals**: layers are grouped into ~6-9 blocks, with standard residual accumulation within each block and cross-block softmax attention over block-level representations. This keeps overhead minimal while capturing most of the benefit.

```
Input Tokens
    |
Token Embedding (b0)
    |
[AttnRes Block 1: layers 0-1]
    |  BlockAttnRes -> RMSNorm -> GQA Attention + RoPE -> accumulate
    |  BlockAttnRes -> RMSNorm -> SwiGLU FFN -> accumulate
    |  ... (repeat for layers in block)
    |  -> commit block representation
    |
[AttnRes Block 2: layers 2-3]
    |  (same, now attends over b0 + block1)
    ...
[AttnRes Block N]
    |
Final BlockAttnRes (over all blocks)
    |
RMSNorm -> Linear Head (tied weights) -> Logits
```

## Quick Start

```bash
pip install -r requirements.txt

# Train on Shakespeare (~5 min on GPU)
python train.py --data data/shakespeare.txt --preset small

# Generate text
python inference.py --checkpoint checkpoints/best.pt --prompt "To be, or not to be"
```

## Training

### Local file

```bash
python train.py --data data/shakespeare.txt --preset small --max_steps 5000
```

### HuggingFace streaming (OpenWebText + FineWeb-Edu + SlimPajama)

Streams directly from HuggingFace with zero disk usage. Requires internet.

```bash
pip install datasets
python train.py --data hf --preset small --max_steps 50000
```

The three datasets are mixed with default weights: 50% FineWeb-Edu (educational web text), 30% SlimPajama (web + books + code), 20% OpenWebText (Reddit-curated web).

### Instruction tuning (SFT)

After pretraining, fine-tune a checkpoint on a HuggingFace instruction dataset:

```bash
python sft.py --checkpoint checkpoints/best.pt --dataset tatsu-lab/alpaca \
    --out_dir sft_checkpoints --max_steps 3000 --batch_size 4 --max_lr 2e-5
```

Better built-in SFT dataset presets are available:

```bash
# Cleaner small instruction dataset
python sft.py --checkpoint checkpoints/best.pt --preset dolly \
    --out_dir sft_checkpoints_dolly --max_steps 1500 --batch_size 4 --max_lr 2e-5

# Larger chat-style dataset
python sft.py --checkpoint checkpoints/best.pt --preset ultrachat \
    --out_dir sft_checkpoints_ultrachat --max_examples 50000 --max_steps 5000 --batch_size 4 --max_lr 2e-5

# Reconstructed user/assistant pairs from OpenAssistant
python sft.py --checkpoint checkpoints/best.pt --preset openassistant \
    --out_dir sft_checkpoints_oasst --max_steps 3000 --batch_size 4 --max_lr 2e-5
```

The SFT script formats examples as:

```text
User: <instruction plus optional input>
Assistant: <response>
```

and masks prompt tokens so loss is applied only to the assistant response. Use `--preset {alpaca,dolly,ultrachat,openassistant}` for common datasets, or `--preset custom` with `--instruction_field`, `--input_field`, and `--response_field` for other Alpaca-style datasets.

### Key training options

```
--preset {small,medium,large,xl}   Model size (see table below)
--data PATH_OR_hf                  Text file path, or 'hf' for HuggingFace streaming
--batch_size N                     Micro batch size (auto-detected for your GPU if omitted)
--grad_accum_steps N               Gradient accumulation steps (default: 4)
--max_steps N                      Total training steps (default: 5000)
--max_lr FLOAT                     Peak learning rate (default: 3e-4)
--dropout FLOAT                    Dropout rate, 0.1 recommended for small datasets (default: 0.0)
--compile                          Enable torch.compile (can speed up training on GPU)
--out_dir DIR                      Checkpoint and metrics output directory (default: checkpoints/)
```

### Metrics

Training metrics are saved to `checkpoints/metrics.csv` with columns: `step, train_loss, train_ppl, val_loss, val_ppl, lr, epoch, tok_per_sec, vram_gb, elapsed_sec`.

Plot training curves:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("checkpoints/metrics.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df["step"], df["train_loss"], label="train")
val = df.dropna(subset=["val_loss"])
ax1.plot(val["step"], val["val_loss"], label="val", marker="o")
ax1.set(xlabel="step", ylabel="loss", title="Loss")
ax1.legend()

ax2.plot(df.dropna(subset=["tok_per_sec"])["step"],
         df.dropna(subset=["tok_per_sec"])["tok_per_sec"])
ax2.set(xlabel="step", ylabel="tok/s", title="Throughput")

plt.tight_layout()
plt.savefig("checkpoints/curves.png")
plt.show()
```

## Inference

```bash
python inference.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" \
    --max_tokens 256 --temperature 0.8 --top_k 50
```

Text streams to the terminal token by token. Add `--compile` for faster generation on GPU.

For an instruction-tuned checkpoint, use the same chat prompt template used during SFT:

```bash
python inference.py --checkpoint sft_checkpoints/best.pt --chat \
    --prompt "Explain what a neural network is." --max_tokens 256
```

## Model Presets

| Preset | Params | d_model | Layers | Heads | KV Heads | AttnRes Blocks | Notes |
|--------|--------|---------|--------|-------|----------|----------------|-------|
| small  | 114M   | 768     | 12     | 12    | 4        | 6              | Fast iteration |
| medium | 322M   | 1024    | 24     | 16    | 4        | 8              | Sweet spot for 32GB GPU |
| large  | 702M   | 1280    | 36     | 20    | 4        | 9              | Needs gradient checkpointing |
| xl     | 1.2B   | 2048    | 24     | 16    | 4        | 8              | All optimizations required |

All presets use context length 1024, GPT-2 BPE tokenizer (50,257 vocab), no bias, and weight tying.

## Project Structure

```
tiny-gpt/
    config.py          GPTConfig + TrainConfig dataclasses with size presets
    model.py           RMSNorm, RoPE, GQA attention, SwiGLU FFN, BlockAttnResOp, GPT
    train.py           Training loop with BF16, fused AdamW, cosine LR, prefetching
    inference.py       Autoregressive text generation with sampling
    metrics.py         CSV logger for training metrics
    requirements.txt   torch, tiktoken, tqdm, datasets
    data/
        __init__.py
        text.py        TextDataset -- load and tokenize a local text file
        streaming.py   StreamingMixDataset -- stream from HuggingFace with weighted mixing
        registry.py    HuggingFace dataset configs (OpenWebText, FineWeb-Edu, SlimPajama)
        shakespeare.txt
```

## GPU Optimizations

- **BF16 mixed precision** via `torch.amp.autocast`
- **TF32 matmul** via `torch.set_float32_matmul_precision("high")`
- **Flash Attention** via `F.scaled_dot_product_attention` with native GQA
- **Fused AdamW** optimizer (single kernel per step)
- **CUDA stream prefetching** (overlaps data transfer with compute)
- **Gradient checkpointing** for large/xl presets
- **`torch.compile`** support (use `--compile` flag)
- **Auto batch size** detection based on available VRAM

## References

- [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) -- Moonshot AI / Kimi Team
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) -- Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- Radford et al. (GPT-2)
