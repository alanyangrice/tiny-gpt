from __future__ import annotations

from dataclasses import dataclass

# Micro-batch sizes by model preset, keyed by minimum total VRAM (GiB) of the CUDA device.
# Tiers are checked from top to bottom; the first tier with total_vram_gib >= min_gib wins.
# Block AttnRes is memory-heavy; medium+ presets use gradient checkpointing in GPTConfig.
AUTO_MICRO_BATCH_TIERS: tuple[tuple[float, dict[str, int]], ...] = (
    (30.0, {"small": 8, "medium": 12, "large": 4, "xl": 2}),
    (22.0, {"small": 8, "medium": 8, "large": 3, "xl": 1}),
    (0.0, {"small": 4, "medium": 3, "large": 1, "xl": 1}),
)

# When training on CPU/MPS, cap auto micro-batch after VRAM lookup (vram_gib is 0 → lowest tier).
AUTO_MICRO_BATCH_CPU_MAX = 4


def auto_micro_batch_size(total_vram_gib: float, preset: str) -> int:
    """Return default micro-batch size for ``preset`` given GPU total memory in GiB (0 if not CUDA)."""
    for min_gib, table in AUTO_MICRO_BATCH_TIERS:
        if total_vram_gib >= min_gib:
            return table[preset]
    raise RuntimeError("AUTO_MICRO_BATCH_TIERS must be non-empty")


def _round_multiple(x: int, multiple: int) -> int:
    return multiple * ((x + multiple - 1) // multiple)


@dataclass
class GPTConfig:
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0

    num_attn_res_blocks: int = 6

    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        assert self.n_layer % self.num_attn_res_blocks == 0, (
            f"n_layer ({self.n_layer}) must be divisible by "
            f"num_attn_res_blocks ({self.num_attn_res_blocks})"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def d_ff(self) -> int:
        return _round_multiple(int(8 / 3 * self.d_model), 256)

    @property
    def layers_per_attn_res_block(self) -> int:
        """Number of transformer layers (attn+mlp pairs) in each AttnRes block."""
        return self.n_layer // self.num_attn_res_blocks

    @staticmethod
    def small() -> "GPTConfig":
        """~114M params. Fits easily on 32 GB with large batch sizes."""
        return GPTConfig(
            d_model=768, n_layer=12, n_head=12, n_kv_head=4,
            block_size=1024, num_attn_res_blocks=6,
        )

    @staticmethod
    def medium() -> "GPTConfig":
        """~350M params. Tuned for ~32 GB GPUs (e.g. RTX 5090).

        Block AttnRes keeps every committed block alive for the forward pass, so
        activation memory is much higher than a vanilla transformer of the same
        width/depth. Gradient checkpointing is enabled to avoid VRAM overspill
        into unified/host memory on Windows.
        """
        return GPTConfig(
            d_model=1024, n_layer=24, n_head=16, n_kv_head=4,
            block_size=1024, num_attn_res_blocks=8,
            use_gradient_checkpointing=True,
        )

    @staticmethod
    def large() -> "GPTConfig":
        """~770M params. Needs gradient checkpointing, batch size 2-4."""
        return GPTConfig(
            d_model=1280, n_layer=36, n_head=20, n_kv_head=4,
            block_size=1024, num_attn_res_blocks=9,
            use_gradient_checkpointing=True,
        )

    @staticmethod
    def xl() -> "GPTConfig":
        """~1.5B params. Aggressive, all optimizations required."""
        return GPTConfig(
            d_model=2048, n_layer=24, n_head=16, n_kv_head=4,
            block_size=1024, num_attn_res_blocks=8,
            use_gradient_checkpointing=True,
        )


@dataclass
class TrainConfig:
    data: str
    preset: str = "small"
    out_dir: str = "checkpoints"
    batch_size: int | None = None
    grad_accum_steps: int = 4
    max_steps: int = 5000
    warmup_steps: int = 200
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.0
    eval_interval: int = 250
    eval_steps: int = 20
    compile: bool = False
    device: str | None = None
