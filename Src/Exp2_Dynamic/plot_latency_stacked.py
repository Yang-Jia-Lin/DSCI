"""
Src/Exp2_Dynamic/plot_latency_stacked.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_latency_stacked(user_labels, T_parts, save_path: Path):
    T1, T2, T3, T4, T5 = T_parts
    x = np.arange(len(user_labels))
    bottom = np.zeros_like(x, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for comp, lab in zip(
        [T1, T2, T3, T4, T5],
        ["T1 Local", "T2 U→E Tx", "T3 Edge Comp", "T4 E→C Tx", "T5 Cloud Comp"],
    ):
        ax.bar(x, comp, bottom=bottom, label=lab)
        bottom += comp

    ax.set_xticks(x)
    ax.set_xticklabels(user_labels)
    ax.set_ylabel("Latency")
    ax.set_title("Latency Composition (Stacked)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=len(user_labels), fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

