from dataclasses import dataclass


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
        """~350M params. Sweet spot for RTX 5090."""
        return GPTConfig(
            d_model=1024, n_layer=24, n_head=16, n_kv_head=4,
            block_size=1024, num_attn_res_blocks=8,
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
