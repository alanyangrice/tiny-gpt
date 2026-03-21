"""Registry of HuggingFace datasets available for streaming pre-training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HFDatasetConfig:
    name: str
    path: str
    config_name: str | None = None
    split: str = "train"
    text_field: str = "text"
    weight: float = 1.0


DATASET_PRESETS: dict[str, HFDatasetConfig] = {
    "openwebtext": HFDatasetConfig(
        name="OpenWebText",
        path="Skylion007/openwebtext",
        weight=0.2,
    ),
    "fineweb_edu": HFDatasetConfig(
        name="FineWeb-Edu-10B",
        path="HuggingFaceFW/fineweb-edu",
        config_name="sample-10BT",
        weight=0.5,
    ),
    "slimpajama": HFDatasetConfig(
        name="SlimPajama-627B",
        path="cerebras/SlimPajama-627B",
        weight=0.3,
    ),
}
