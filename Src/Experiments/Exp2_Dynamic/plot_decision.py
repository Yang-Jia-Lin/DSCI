"""
Src/Exp2_Dynamic/plot_decision.py
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from Src.paras import EARLY_EXIT_LAYERS


def plot_XY(X_opt, Y_opt, data_name: str, save_dir: Path):
    stem = Path(data_name).stem
    n, m = X_opt.shape
    E = EARLY_EXIT_LAYERS
    Y_E = Y_opt[:, E]

    # ---- X heatmap ----
    figX, axX = plt.subplots(dpi=300)
    axX.set_xlabel("DNN lauers", fontsize=16)  # 设置X轴标签字体大小
    axX.set_ylabel("users", fontsize=16)  # 设置Y轴标签字体大小
    axX.set_title("X decision", fontsize=18)  # 设置标题字体大小

    # 设置 X 轴刻度位置和标签
    axX.set_xticks(np.arange(m))  # 设置刻度位置为0到m-1
    axX.set_xticklabels([i + 1 for i in range(m)], fontsize=14)  # 设置刻度标签为 1, 2, 3, ..., m
    axX.set_yticks(range(n))  # 设置Y轴刻度位置
    axX.set_yticklabels([i + 1 for i in range(n)], fontsize=14)  # 设置Y轴刻度标签为 1, 2, 3, ..., n

    axX.matshow(X_opt, cmap='Blues', aspect='auto')  # 显示 X 的热力图
    for l in E:  # 早退层分割线
        axX.axvline(x=l, color="red", linestyle="--", linewidth=0.8)
    axX.tick_params(labelsize=14)  # 设置刻度标签字体大小
    axX.xaxis.set_ticks_position('bottom')  # 确保X轴刻度在底部

    # ---- Y heatmap ----
    figY, axY = plt.subplots(dpi=300)
    im_Y = axY.imshow(Y_E, aspect="auto", cmap="viridis", interpolation="nearest", origin='lower')
    axY.set_xlabel("DNN layers", fontsize=16)  # 设置X轴标签字体大小
    axY.set_ylabel("users", fontsize=16)  # 设置Y轴标签字体大小
    axY.set_title("Y decision", fontsize=18)

    axY.set_xticks(range(len(E)))  # 设置X轴刻度位置为早退层
    axY.set_xticklabels(E, fontsize=14)  # 设置X轴刻度标签为早退层的索引
    axY.set_yticks(range(n))  # 设置Y轴刻度位置
    axY.set_yticklabels([i + 1 for i in range(n)], fontsize=14)  # 设置Y轴刻度标签为 1, 2, 3, ..., n

    axY.tick_params(labelsize=14)  # 设置刻度标签字体大小
    axY.xaxis.set_ticks_position('bottom')  # 确保X轴刻度在底部

    # 设置 Y 轴从上到下增大
    axY.invert_yaxis()  # 反转Y轴，确保从上到下增大

    # Add color bar for Y
    cbar_Y = figY.colorbar(im_Y, ax=axY, fraction=0.046)
    cbar_Y.set_label("threshold", fontsize=14)  # 设置颜色条标签字体大小
    figY.tight_layout()
    plt.show()

    # ---- Save results ----
    if save_dir is not None:
        save_path_X = save_dir / f"{stem}_X_{datetime.now().strftime('%m%d_%H%M')}.svg"
        figX.savefig(save_path_X, format='svg')
        print(f"结果X已保存至: {save_path_X}")
        save_path_Y = save_dir / f"{stem}_Y_{datetime.now().strftime('%m%d_%H%M')}.svg"
        figY.savefig(save_path_Y, format='svg')
        print(f"结果Y已保存至: {save_path_Y}")


def plot_X(user_labels, cut_points: np.ndarray, save_path: Path):
    x = np.arange(len(user_labels))
    cut0 = cut_points[:, 0]
    cut1 = cut_points[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, cut0, marker="o", label="cut0 (Device→Edge)")
    mask = cut1 >= 0
    ax.scatter(x[mask], cut1[mask], marker="^", label="cut1 (Edge→Cloud)")

    ax.set_xticks(x)
    ax.set_xticklabels(user_labels)
    ax.set_ylabel("Layer Index")
    ax.set_title("Model Partition Points")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    from Src.paras import Paras, RESULT_TEST_PATH
    paras = Paras()
    X = np.zeros([paras.n, paras.m])
    X[0, [10, 20]] = 1
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