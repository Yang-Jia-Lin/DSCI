"""
Src/Exp2_Dynamic/plot_latency_stacked.py
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_latency_stacked(user_labels, T_parts, save_dir: Path):
    """
    绘制延迟组成的堆叠柱状图
    """
    set_ieee_style(mode='single')
    plt.rcParams['figure.figsize'] = (4.0, 3.5)
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 0.5, 5))
    x = np.arange(len(user_labels))
    bottom = np.zeros_like(x, dtype=float)
    labels = ["Local", "U$\\to$E", "Edge", "E$\\to$C", "Cloud"]
    for comp, lab, col in zip(T_parts, labels, colors):
        ax.bar(x, comp,
               bottom=bottom,
               label=lab,
               color=col,
               edgecolor='white',
               linewidth=0.5,
               width=0.8)
        bottom += comp
    ax.set_xticks(x)
    ax.set_xticklabels(user_labels)
    ax.set_xlabel("User Index")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Composition")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center',
              fontsize=10,
              bbox_to_anchor=(0.5, 1.08),
              ncol=3,
              frameon=False,
              columnspacing=1.0,
              handletextpad=0.4)
    plt.tight_layout(pad=0.15)

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"Latency_Stacked_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


if __name__ == "__main__":
    users = [f"U{i + 1}" for i in range(5)]
    T = [np.random.rand(5) * 0.2 for _ in range(5)]

    from Src.paras import RESULT_TEST_PATH
    plot_latency_stacked(users, T, save_dir = Path(RESULT_TEST_PATH))

