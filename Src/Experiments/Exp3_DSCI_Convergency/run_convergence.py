"""
Src/Experiments/Exp3_DSCI_Convergency/run_convergence.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from Src.paras import RESULT_CONVERGENCE_PATH
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def load_metrics_from_exp(exp_dir: Path) -> pd.DataFrame:
    """从具体的实验文件夹中加载 metrics.jsonl 数据"""
    metrics_path = exp_dir / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"Error: {exp_dir} 中未找到 metrics.jsonl")
        return pd.DataFrame()

    data = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def run_convergence_analysis(exp_path: Path, save_root: Path = RESULT_CONVERGENCE_PATH, save_results: bool = True):
    exp_path = Path(exp_path)
    df = load_metrics_from_exp(exp_path)
    set_ieee_style(mode='single')

    # 设置保存路径
    fig_save_dir = Path(save_root) / exp_path.name
    if save_results:
        fig_save_dir.mkdir(parents=True, exist_ok=True)

    algo_label = exp_path.name.split('_')[0]

    # ======== 总效用收敛曲线 (outer_obj) ========
    plt.figure()
    if 'outer_obj' in df.columns:
        plt.plot(df['outer_obj'], label='Total Utility')

    plt.xlabel('Epoch')
    plt.ylabel('Utility')
    plt.title(f'{algo_label} Convergence')
    plt.legend()

    if save_results:
        save_fig_for_ieee(fig_save_dir / "utility_convergence")
    plt.show()

    # ======== 熵收敛曲线 (Entropy X & Y) ========
    plt.figure()
    if 'entropy_X' in df.columns:
        plt.plot(df['entropy_X'], label='Entropy $\mathbf{X}$')
    if 'entropy_Y' in df.columns:
        plt.plot(df['entropy_Y'], label='Entropy $\mathbf{Y}$', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.legend()
    if save_results:
        save_fig_for_ieee(fig_save_dir / "entropy_convergence")
    plt.show()

    # ======== 性能指标变化 (Latency & Accuracy) ========
    color_lat = '#d62728'  # 红色系
    color_acc = '#2ca02c'  # 绿色系
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    l1, = ax1.plot(df['latency'], '--', color = color_lat,label="Latency")
    ax1.set_ylabel('Latency (s)')
    ax1.tick_params(axis='y')

    l2, = ax2.plot(df['acc'], '-', color = color_acc,label="Accuracy")
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')

    ax1.set_xlabel('Epoch')
    lines = [l1, l2]
    ax1.legend(lines, [line.get_label() for line in lines],
               loc='upper center',  # 以图例的顶部中心为对齐点
               bbox_to_anchor=(0.5, 1.18),  # (x, y) 坐标，1.15 表示在坐标轴上方 15% 处
               ncol=2,  # 设置为 2 列，横向排列更节省空间
               frameon=False)
    if save_results:
        save_fig_for_ieee(fig_save_dir / "perf_tradeoff")
    plt.show()


if __name__ == "__main__":
    target_path = Path(r"D:\Coding\Python\DSCI\Result\Optimize\PPO\PPO_20260126_104604")
    run_convergence_analysis(target_path)