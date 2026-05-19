from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "checkpoints" / "metrics.csv"
OUTPUT_PATH = ROOT / "assets" / "training_results.png"


def maybe_float(value: str) -> float | None:
    return float(value) if value else None


def read_metrics(path: Path) -> list[dict[str, float | None]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [{key: maybe_float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def plot_series(
    ax,
    rows: Iterable[dict[str, float | None]],
    key: str,
    label: str,
    color: str,
    *,
    marker: str | None = None,
) -> None:
    points = [(row["step"], row[key]) for row in rows if row[key] is not None]
    steps = [point[0] for point in points]
    values = [point[1] for point in points]
    ax.plot(steps, values, label=label, color=color, linewidth=2.5, marker=marker, markersize=5)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(METRICS_PATH)
    val_rows = [row for row in rows if row["val_loss"] is not None]
    best_val = min(val_rows, key=lambda row: row["val_loss"])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor("white")

    ax.set_facecolor("white")
    ax.grid(True, color="#d1d5db", alpha=0.7, linewidth=0.8)
    ax.tick_params(colors="#374151", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")

    plot_series(ax, rows, "train_loss", "training loss", "#2563eb")
    plot_series(ax, rows, "val_loss", "validation loss", "#dc2626", marker="o")
    ax.scatter(best_val["step"], best_val["val_loss"], color="#f59e0b", s=90, zorder=5, label="best result")

    ax.set_title("TinyGPT Pretraining Results", color="#111827", loc="left", fontsize=24, fontweight="bold", pad=28)
    ax.set_xlabel("training step", color="#374151", fontsize=11)
    ax.set_ylabel("cross entropy loss", color="#374151", fontsize=11)
    ax.legend(facecolor="white", edgecolor="#d1d5db", labelcolor="#111827", fontsize=10)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
