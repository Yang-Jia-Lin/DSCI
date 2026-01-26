"""
Src/Exp2_Dynamic/plot_decision.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Src.paras import EARLY_EXIT_LAYERS
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


if __name__ == "__main__":
    from Src.paras import Paras, RESULT_TEST_PATH
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
    F_e = np.ones((paras.n, 1), dtype=np.float32) * (paras.f_e_max / paras.n)
    F_c = np.ones((paras.n, 1), dtype=np.float32) * (paras.f_c_max / paras.n)
    # test 1
    plot_XY(X, Y, data_name="Test_plot_XY.svg", save_dir=Path(RESULT_TEST_PATH))