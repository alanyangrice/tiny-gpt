"""
Simple CSV metrics logger for training runs.

Writes a CSV file to the output directory so training curves can be
plotted with any tool (matplotlib, pandas, Excel, etc.).

Example usage after training:

    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv("checkpoints/metrics.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df["step"], df["train_loss"], label="train")
    ax1.plot(df.dropna(subset=["val_loss"])["step"],
             df.dropna(subset=["val_loss"])["val_loss"], label="val")
    ax1.set(xlabel="step", ylabel="loss"); ax1.legend()
    ax2.plot(df["step"], df["tok_per_sec"])
    ax2.set(xlabel="step", ylabel="tok/s")
    plt.tight_layout(); plt.savefig("checkpoints/curves.png"); plt.show()
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

FIELDNAMES = [
    "step",
    "train_loss",
    "train_ppl",
    "val_loss",
    "val_ppl",
    "lr",
    "epoch",
    "tok_per_sec",
    "vram_gb",
    "elapsed_sec",
]


class MetricsLogger:
    """Append-only CSV writer. Each call to ``log`` writes one row."""

    def __init__(self, out_dir: str, filename: str = "metrics.csv"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = Path(out_dir) / filename
        self._fp = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=FIELDNAMES, extrasaction="ignore")
        self._writer.writeheader()
        self._fp.flush()

    def log(self, **kwargs) -> None:
        self._writer.writerow(kwargs)
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()
