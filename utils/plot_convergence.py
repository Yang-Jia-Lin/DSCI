from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_convergence(history, data_name: str = "Convergence",save_dir: Path = None):
    history = [1.0 * x for x in history]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('目标函数值', fontsize=12)
    plt.title('收敛曲线', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}..png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 高分辨率保存
        print(f"收敛曲线已保存至: {save_path}")

    plt.show()


def save_thr_data(
        history_data: list[int],
        data_name: str,
        save_dir: Path
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}.csv"
    df = pd.DataFrame({'values': history_data})
    csv_path = save_dir / filename
    df.to_csv(csv_path, index=False)
    return csv_path