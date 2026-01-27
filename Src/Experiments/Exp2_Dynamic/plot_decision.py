"""
Src/Exp2_Dynamic/plot_decision.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from Src.paras import EARLY_EXIT_LAYERS, Paras, RESULT_TEST_PATH
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_XY(X_opt, Y_opt, data_name: str, save_dir: Path):
    """
    绘制决策变量 X 和 Y 的热力图
    """
    stem = Path(data_name).stem
    n, m = X_opt.shape
    E = EARLY_EXIT_LAYERS
    Y_E = Y_opt[:, E]
    set_ieee_style(mode='single')

    # ---- 绘制 X (模型划分决策) ----
    figX, axX = plt.subplots()
    axX.matshow(X_opt, cmap='Blues', aspect='auto')
    axX.set_xlabel("DNN Layers")
    axX.set_ylabel("Users")
    axX.set_title("Partitioning Decision ($\mathbf{X}$)")

    # 设置刻度
    axX.set_xticks(np.arange(0, m, max(1, m // 10)))
    axX.set_xticklabels([i + 1 for i in range(0, m, max(1, m // 10))])
    axX.set_yticks(np.arange(0, n, max(1, n // 5)))
    axX.set_yticklabels([i + 1 for i in range(0, n, max(1, n // 5))])

    # 绘制早退层分割线
    for l in E:
        axX.axvline(x=l, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    axX.xaxis.set_ticks_position('bottom')  # 确保 X 轴刻度在底部

    # ---- 绘制 Y (早退阈值决策) ----
    figY, axY = plt.subplots()
    # Y 是连续的阈值，需要 colorbar
    im_Y = axY.imshow(Y_E, aspect="auto", cmap="viridis", interpolation="nearest")

    axY.set_xlabel("Early Exit Layers")
    axY.set_ylabel("Users")
    axY.set_title("Threshold Decision ($\mathbf{Y}$)")

    axY.set_xticks(range(len(E)))
    axY.set_xticklabels(E)
    axY.set_yticks(np.arange(0, n, max(1, n // 5)))
    axY.set_yticklabels([i + 1 for i in range(0, n, max(1, n // 5))])

    axY.xaxis.set_ticks_position('bottom')
    axY.invert_yaxis()

    cbar_Y = figY.colorbar(im_Y, ax=axY, fraction=0.046, pad=0.04)
    cbar_Y.set_label("Threshold $\epsilon$")

    # ---- 保存结果 ----
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_for_ieee(save_dir / f"{stem}_X")
        save_fig_for_ieee(save_dir / f"{stem}_Y")
        print(f"决策热力图已保存至: {save_dir}")
    plt.show()


def plot_X(user_labels, cut_points: np.ndarray, save_path: Path):
    """
    绘制模型切分点散点图
    """
    save_path = Path(save_path)
    set_ieee_style(mode='single')

    x = np.arange(len(user_labels))
    cut0 = cut_points[:, 0]
    cut1 = cut_points[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x, cut0, marker="o", label="Device $\\to$ Edge", alpha=0.8)

    mask = cut1 >= 0
    if np.any(mask):
        ax.scatter(x[mask], cut1[mask], marker="^", label="Edge $\\to$ Cloud", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(user_labels)
    ax.set_ylabel("Layer Index")
    ax.set_xlabel("User Index")
    ax.set_title("Model Partition Points")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True)

    save_fig_for_ieee(save_path)
    print(f"切分点散点图已保存至: {save_path.with_suffix('.pdf')}")
    plt.show()


def plot_resource_trend(csv_path: Path, save_dir: Path, total_layers: int = 128):
    """
    读取动态实验 CSV 并生成三色空间划分趋势图
    """
    # 1. 加载数据
    if not csv_path.exists():
        print(f"[Error] CSV file not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    f_u = df["F_u"].values
    cut_ee = df["avg_end_edge"].values  # End-Edge 边界
    cut_ec = df["avg_edge_cloud"].values  # Edge-Cloud 边界
    utility = df["total_utility"].values

    # 2. 设置 IEEE 绘图风格
    set_ieee_style(mode='single')
    plt.rcParams['figure.figsize'] = (4.0, 3.3)
    fig, ax1 = plt.subplots()

    # 3. 绘制区域填充 (三色空间划分)
    # 区间1: Local Processing (0 到 End-Edge)
    ax1.fill_between(f_u, 0, cut_ee, color='#DAE8FC', alpha=0.8, label='Local')
    # 区间2: Edge Processing (End-Edge 到 Edge-Cloud)
    ax1.fill_between(f_u, cut_ee, cut_ec, color='#D5E8D4', alpha=0.8, label='Edge')
    # 区间3: Cloud Processing (Edge-Cloud 到 Total Layers)
    ax1.fill_between(f_u, cut_ec, total_layers, color='#FFE6CC', alpha=0.8, label='Cloud')

    # 4. 绘制边界线
    ax1.plot(f_u, cut_ee, color='#6C8EBF', linestyle='-', marker='o', markersize=4, linewidth=1.5)
    ax1.plot(f_u, cut_ec, color='#82B366', linestyle='-', marker='s', markersize=4, linewidth=1.5)

    # 5. 设置主轴 (层数)
    ax1.set_xlabel('User Device Frequency $F_u$ (GHz)')
    ax1.set_ylabel('DNN Layer Index')
    ax1.set_ylim(0, total_layers)
    ax1.set_xlim(f_u.min(), f_u.max())

    # 6. 绘制次轴 (总效用) - 可选，展示效用随算力的增长
    ax2 = ax1.twinx()
    ax2.plot(f_u, utility, color='#B85450', linestyle='--', linewidth=1.2, label='Total Utility')
    ax2.set_ylabel('Total System Utility')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center',  bbox_to_anchor=(0.5, 1.01), fontsize='small', frameon=True, ncol=2)
    # ax1.set_title("Decision Sensitivity vs. User Capacity")
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # 7. 保存
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_fig_for_ieee(save_dir / "resource_trend_analysis")
        print(f"[Success] Trend plot saved to: {save_dir / 'resource_trend_analysis.pdf'}")

    plt.show()




# ==============================
# =========== TEST =============
# ==============================

def test_resource_trend():
    # 1. 定义路径
    test_dir = Path(RESULT_TEST_PATH) / "Test_Resource_Trend"
    test_csv = test_dir / "test_dynamic_data.csv"

    # 2. 生成模拟数据
    f_u_range = np.arange(0.5, 8.5, 0.5)
    n_steps = len(f_u_range)

    # 3. 构造模拟趋势：
    # 3.1 End-Edge: 随算力增加从第 10 层线性增加到第 80 层左右
    avg_end_edge = 10 + 8 * f_u_range + np.random.normal(0, 2, n_steps)
    # 3.2 Edge-Cloud: 相对稳定，在大约 100 层左右波动
    avg_edge_cloud = 90 + 3 * f_u_range + np.random.normal(0, 2, n_steps)
    # 3.3 Utility: 收益递减的增长曲线
    total_utility = 500 + 200 * np.log1p(f_u_range)
    avg_end_edge = np.clip(avg_end_edge, 0, 120)
    avg_edge_cloud = np.clip(avg_edge_cloud, avg_end_edge + 5, 127)
    # 3.4 保存csv用于测试
    df = pd.DataFrame({
        "F_u": f_u_range,
        "avg_end_edge": avg_end_edge,
        "avg_edge_cloud": avg_edge_cloud,
        "total_utility": total_utility
    })
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_csv, index=False)
    print(f"[Test] Mock CSV generated at: {test_csv}")

    # 4. 执行绘图
    print("[Test] Starting plot_resource_trend...")
    plot_resource_trend(
        csv_path=test_csv,
        save_dir=test_dir,
        total_layers=128
    )
    print(f"[Test] Plotting complete. Check results in: {test_dir}")


def test_XY():
    paras = Paras()
    X = np.zeros([paras.n, paras.m])
    X[0, [10, 20]] = 1
    X[1, [40, 60]] = 1
    Y = np.ones([paras.n, paras.m])
    Y[2, 57] = 0.6
    Y[4, 103] = 0.8
    Y[5, 57] = 0.5
    Y[5, 103] = 0.9
    Y[7, 57] = 0.1
    Y[9, 103] = 0.5
    # F_e = np.ones((paras.n, 1), dtype=np.float32) * (paras.f_e_max / paras.n)
    # F_c = np.ones((paras.n, 1), dtype=np.float32) * (paras.f_c_max / paras.n)
    # test 1
    plot_XY(X, Y, data_name="Test_plot_XY.svg", save_dir=Path(RESULT_TEST_PATH))


if __name__ == "__main__":
    # test_XY()
    test_resource_trend()