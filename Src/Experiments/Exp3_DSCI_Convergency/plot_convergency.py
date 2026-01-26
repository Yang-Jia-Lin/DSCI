"""
Src/Exp3_DSCI_Convergency/plot_convergency.py
"""
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def plot_convergence(history, data_name: str = "Convergence", save_dir: Path = None):
    history = [1.0 * x for x in history]
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel('iteration', fontsize=16)
    plt.ylabel('objective', fontsize=16)
    plt.title('Convergence', fontsize=18, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}.svg"
        plt.savefig(save_path, format='svg')
        print(f"收敛曲线已保存至: {save_path}")

    plt.show()
