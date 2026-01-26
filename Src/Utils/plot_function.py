"""
Src/Utils/log_function.py
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from Src.Objective.compute_latency import compute_5_latency
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.objective import objective
from Src.paras import RESULT_DYNAMIC_PATH, EARLY_EXIT_LAYERS
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_convergence(history, data_name: str = "Convergence", save_dir: Path = None):
    history = [1.0 * x for x in history]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel('迭代次数', fontsize=16)
    plt.ylabel('目标函数值', fontsize=16)
    plt.title('收敛曲线', fontsize=18, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}.svg"
        plt.savefig(save_path, format='svg')
        print(f"收敛曲线已保存至: {save_path}")

    plt.show()


def plot_decisions(X_opt, Y_opt, data_name: str, save_dir: Path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    stem = Path(data_name).stem

    n, m = X_opt.shape
    E = EARLY_EXIT_LAYERS
    Y_E = Y_opt[:, E]

    # ---- X heatmap ----
    figX, axX = plt.subplots(dpi=300)
    axX.set_xlabel("DNN层", fontsize=16)  # 设置X轴标签字体大小
    axX.set_ylabel("用户", fontsize=16)  # 设置Y轴标签字体大小
    axX.set_title("切分决策 (X)", fontsize=18)  # 设置标题字体大小

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
    axY.set_xlabel("DNN层", fontsize=16)  # 设置X轴标签字体大小
    axY.set_ylabel("用户", fontsize=16)  # 设置Y轴标签字体大小
    axY.set_title("早退阈值决策 (Y)", fontsize=18)

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
    cbar_Y.set_label("阈值", fontsize=14)  # 设置颜色条标签字体大小
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


def _parse_cut_points_from_X(X: np.ndarray) -> np.ndarray:
    """
    将 0-1 矩阵 X 解析为每个用户的切分点对 (cut0, cut1)。
    默认约定：每行出现的前两个 1 的位置分别是 cut0/cut1。
    若只有 1 个 1，则 cut1=-1；若没有 1，则 (-1,-1)。
    """
    X = np.asarray(X)
    n, _ = X.shape
    cuts = np.full((n, 2), -1, dtype=int)
    for i in range(n):
        idx = np.where(X[i] == 1)[0]
        if idx.size >= 2:
            cuts[i] = [int(idx[0]), int(idx[1])]
        elif idx.size == 1:
            cuts[i] = [int(idx[0]), -1]
    return cuts


def _plot_latency_stacked(user_labels, T_parts, save_path: Path):
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


def _plot_cut_points(user_labels, cut_points: np.ndarray, save_path: Path):
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


def plot_user_dynamic(X, Y, F_e, F_c, paras):
    # ========= 1) 数据准备 ========
    best_val = objective(X, Y, F_e, F_c, paras)
    P = compute_layer_exit_probs(Y, paras)
    T1, T2, T3, T4, T5 = compute_5_latency(X, P, F_e, F_c, paras)
    cut_points = _parse_cut_points_from_X(X)

    # ========= 2) 画图 ========
    user_labels = [str(i + 1) for i in range(paras.n)]
    # user_labels = [r"Low ($F_u=0.1$)", r"Mid ($F_u=2$)", r"High ($F_u=8$)"]
    out_dir = Path(RESULT_DYNAMIC_PATH) / f"UserHetero_{datetime.now().strftime('%m%d_%H%M')}"
    _plot_latency_stacked(
        user_labels,
        (T1, T2, T3, T4, T5),
        out_dir / "latency_stacked.png",
    )
    _plot_cut_points(
        user_labels,
        cut_points,
        out_dir / "cut_points.png",
    )

    # ========= 4) 打印结果 ========
    print(f"[OK] best_val={best_val:.6f}")
    print(f"[Saved] {out_dir / 'latency_stacked.png'}")
    print(f"[Saved] {out_dir / 'cut_points.png'}")


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
    plot_decisions(X, Y, data_name="最优结果示意图.svg", save_dir=Path(RESULT_TEST_PATH))
