"""
Src/Exp2_Dynamic/plot_latency_stacked.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# 1. 导入 IEEE 模板工具
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_latency_stacked(user_labels, T_parts, save_path: Path):
    """
    绘制延迟组成的堆叠柱状图
    """
    save_path = Path(save_path)
    set_ieee_style(mode='single')
    plt.rcParams['figure.figsize'] = (4.0, 4.0)
    T_list = T_parts  # 假设包含 T1, T2, T3, T4, T5
    labels = ["Local", "U$\\to$E", "Edge", "E$\\to$C", "Cloud"]

    # 2. 设置颜色方案
    colors = plt.cm.tab10(np.linspace(0, 0.5, 5))
    x = np.arange(len(user_labels))
    bottom = np.zeros_like(x, dtype=float)
    fig, ax = plt.subplots()

    # 3. 绘制堆叠柱状图
    bar_width = 0.8
    for comp, lab, col in zip(T_list, labels, colors):
        # 添加 width 参数，设置 edgecolor 增加辨识度
        ax.bar(x, comp, bottom=bottom, label=lab, color=col,
               edgecolor='white', linewidth=0.5, width=bar_width)
        bottom += comp

    # 4. 细节设置
    ax.set_xticks(x)
    ax.set_xticklabels(user_labels)
    ax.set_xlabel("User Index")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Composition")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)  # 确保网格在柱子下方

    # 5. 优化图例 (IEEE 双栏建议：图外上方，横向排布)
    ax.legend(loc='lower center',
              fontsize=10,
              bbox_to_anchor=(0.5, 1.08),
              ncol=3,
              frameon=False,
              columnspacing=1.0,
              handletextpad=0.4)

    # 6. 保存逻辑
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_path.with_suffix(''))
    print(f"堆叠柱状图已保存至: {save_path.with_suffix('.pdf')}")
    plt.show()


if __name__ == "__main__":
    # 测试数据
    users = [f"U{i + 1}" for i in range(5)]
    data = [np.random.rand(5) * 0.2 for _ in range(5)]  # 模拟 T1-T5
    from Src.paras import RESULT_TEST_PATH

    test_path = Path(RESULT_TEST_PATH) / "Latency_Stacked_Test"
    plot_latency_stacked(users, data, test_path)

