"""
Src/Dynamic/run_dynamic.py
"""
import matplotlib.pyplot as plt
from datetime import datetime
from Src.paras import *
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Optimizer.PPO.run_PPO import run_dsci_experiment
from Src.Objective.compute_latency import compute_5_latency
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.objective import objective


def parse_cut_points_from_X(X: np.ndarray) -> np.ndarray:
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


def plot_latency_stacked(user_labels, T_parts, save_path: Path):
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
    ax.legend(ncol=x, fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_cut_points(user_labels, cut_points: np.ndarray, save_path: Path):
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


def user_dynamic(solution_dir:Path = None):
    # ========= 1) 定义 paras =========
    n = 10
    F_u = np.array([0.1] * 3 + [2.0] * 4 + [8.0] * 3, dtype=np.float32)          # 低/中/高
    H_u = np.array([2.0] * 10, dtype=np.float32)
    paras = Paras(n=n, F_u=F_u, H_u=H_u,
        m=NUM_LAYERS,
        E=EARLY_EXIT_LAYERS,
        D=DATA_SIZE_LAYERS,
        C=COMPUTE_SIZE_LAYERS,
        f_e_max=float(EDGE_MAX_FREQ),
        f_c_max=float(CLOUD_MAX_FREQ),
        b_e=float(BANDWIDTH_EDGE),
        b_c=float(BANDWIDTH_CLOUD),
        G=float(BASE_STATION_POWER),
        delta=float(NOISE_POWER),
        alpha=1,
        beta=5,
    )
    paras.rates, paras.accs = parsing_rate_and_acc(paras)

    # ========= 2) 跑 DSCI（PPO） =========
    custom_paras_dict = {
        "n": n,
        "F_u": F_u,
        "H_u": H_u,
    }
    if solution_dir is None:
        best_val, best_sol, history = run_dsci_experiment(custom_paras_dict=custom_paras_dict, save_log=True)
        X, Y, F_e, F_c = best_sol
    else:
        npz_path = solution_dir / "solution.npz"
        data = np.load(npz_path, allow_pickle=True)
        X = data['X']
        Y = data['Y']
        F_e = data['F_e']
        F_c = data['F_c']
        best_val = data['best_val']

    # ========= 3) 计算 5 段时延 =========
    P = compute_layer_exit_probs(Y, paras)
    T1, T2, T3, T4, T5 = compute_5_latency(X, P, F_e, F_c, paras)

    # ========= 4) 解析切分点 =========
    cut_points = parse_cut_points_from_X(X)

    # ========= 5) 画图 =========

    out_dir = Path(RESULT_DYNAMIC_PATH) / f"UserHetero_n3_{datetime.now().strftime('%m%d_%H%M')}"
    # user_labels = [r"Low ($F_u=0.1$)", r"Mid ($F_u=2$)", r"High ($F_u=8$)"]
    user_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plot_latency_stacked(
        user_labels,
        (T1, T2, T3, T4, T5),
        out_dir / "latency_stacked.png",
    )
    plot_cut_points(
        user_labels,
        cut_points,
        out_dir / "cut_points.png",
    )

    print(f"[OK] best_val={best_val:.6f}")
    print(f"[Saved] {out_dir / 'latency_stacked.png'}")
    print(f"[Saved] {out_dir / 'cut_points.png'}")



def user_dynamic_test():
    # ========= 1) 定义 paras（n=3，三类端侧算力） =========
    n = 3
    F_u = np.array([0.1, 2.0, 8.0], dtype=np.float32)  # 低/中/高
    H_u = np.array([2.0, 2.0, 2.0], dtype=np.float32)  # 其余保持一致

    paras = Paras(
        n=n,
        m=NUM_LAYERS,
        E=EARLY_EXIT_LAYERS,
        D=DATA_SIZE_LAYERS,
        C=COMPUTE_SIZE_LAYERS,
        F_u=F_u,
        f_e_max=float(EDGE_MAX_FREQ),
        f_c_max=float(CLOUD_MAX_FREQ),
        H_u=H_u,
        b_e=float(BANDWIDTH_EDGE),
        b_c=float(BANDWIDTH_CLOUD),
        G=float(BASE_STATION_POWER),
        delta=float(NOISE_POWER),
        alpha=1,
        beta=5,
    )
    paras.rates, paras.accs = parsing_rate_and_acc(paras)

    # ========= 2) 一个测试解 =========
    X = np.zeros((n, NUM_LAYERS))
    X[0, 20] = 1
    X[0, 50] = 1
    X[1, 10] = 1
    X[1, 100] = 1
    X[2, 0] = 1
    X[2, 1] = 1

    Y = np.ones((n, NUM_LAYERS))
    Y[:, 57] = 0.9
    Y[:, 103] = 0.8

    F_e = np.ones((n, 1)) * (paras.f_e_max / n)
    F_c = np.ones((n, 1)) * (paras.f_c_max / n)
    best_val = objective(X, Y, F_e, F_c, paras)

    # ========= 3) 计算 5 段时延 =========
    P = compute_layer_exit_probs(Y, paras)
    T1, T2, T3, T4, T5 = compute_5_latency(X, P, F_e, F_c, paras)

    # ========= 4) 解析切分点 =========
    cut_points = parse_cut_points_from_X(X)

    # ========= 5) 画图 =========
    out_dir = Path(RESULT_DYNAMIC_PATH) / "UserHetero_test"
    user_labels = [r"Low ($F_u=0.1$)", r"Mid ($F_u=2$)", r"High ($F_u=8$)"]
    plot_latency_stacked(
        user_labels,
        (T1, T2, T3, T4, T5),
        out_dir / "latency_stacked.png",
    )
    plot_cut_points(
        user_labels,
        cut_points,
        out_dir / "cut_points.png",
    )

    print(f"[OK] best_val={best_val:.6f}")
    print(f"[Saved] {out_dir / 'latency_stacked.png'}")
    print(f"[Saved] {out_dir / 'cut_points.png'}")


if __name__ == "__main__":
    # test()
    user_dynamic(solution_dir=Path("D:\Coding\Python\DSCI\Result\PPO\PPO_Exp_20260123_171125"))