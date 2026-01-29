"""
Src/Exp3_DSCI_Convergency/plot_convergency.py
"""
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style
from Src.paras import RESULT_CONVERGENCE_PATH


def plot_convergence(
        utility,
        save_dir: Path = Path(RESULT_CONVERGENCE_PATH),
        alg_name: str = "DSCI"):
    """
    Plot the convergence curve include DSCI、BF、GA
    """
    utility = [1.0 * x for x in utility]

    # 绘图
    set_ieee_style(mode='single')
    plt.figure()
    plt.plot(utility, label=alg_name, linewidth=0.9)
    plt.xlabel('Epoch')
    plt.ylabel('Utility')
    plt.title(f'{alg_name} Convergence')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"{alg_name}_utility_convergence_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


def plot_entropy(entropy_X, entropy_Y, save_dir = Path(RESULT_CONVERGENCE_PATH)):
    plt.figure()
    plt.plot(entropy_X, label='Entropy $\mathbf{X}$', linewidth=1)
    plt.plot(entropy_Y, label='Entropy $\mathbf{Y}$', linestyle='--', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Entropy')
    plt.legend()
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"entropy_convergence_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


def plot_lan_and_acc(latency, acc, save_dir=Path(RESULT_CONVERGENCE_PATH)):
    color_lat = '#d62728'  # 红色系
    color_acc = '#2ca02c'  # 绿色系
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    l1, = ax1.plot(latency, '--', color=color_lat, label="Latency", linewidth=1)
    ax1.set_ylabel('Latency (s)')
    ax1.tick_params(axis='y')

    l2, = ax2.plot(acc, '-', color=color_acc, label="Accuracy", linewidth=1)
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')

    ax1.set_xlabel('Epoch')
    lines = [l1, l2]
    ax1.legend(lines, [line.get_label() for line in lines],
               loc='upper center',  # 以图例的顶部中心为对齐点
               bbox_to_anchor=(0.5, 1.18),  # (x, y) 坐标，1.15 表示在坐标轴上方 15% 处
               ncol=2,  # 设置为 2 列，横向排列更节省空间
               frameon=False)
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"perf_tradeoff_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


if __name__ == '__main__':
    from Src.Utils.log_function import load_and_analyze_results
    PPO_path = Path("D:\Coding\Python\DSCI\Result\Optimize\PPO\PPO_20260127_153244")
    X_opt, Y_opt, F_e, F_c, history, paras = load_and_analyze_results(exp_dir = PPO_path, analysis = False)

    from Src.paras import RESULT_TEST_PATH
    plot_convergence(history, save_dir = Path(RESULT_TEST_PATH))