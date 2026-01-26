"""
Src/Exp2_Dynamic/run_dynamic.py
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from Src.Exp2_Dynamic.plot_decision_XY import plot_X
from Src.Exp2_Dynamic.plot_latency_stacked import plot_latency_stacked
from Src.paras import Paras, RESULT_DYNAMIC_PATH
from Src.Optimizer.PPO.run_PPO import run_dsci_experiment
from Src.Objective.compute_latency import compute_5_latency
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.objective import objective

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
    plot_latency_stacked(
        user_labels,
        (T1, T2, T3, T4, T5),
        out_dir / "latency_stacked.png",
    )
    plot_X(
        user_labels,
        cut_points,
        out_dir / "cut_points.png",
    )

    # ========= 4) 打印结果 ========
    print(f"[OK] best_val={best_val:.6f}")
    print(f"[Saved] {out_dir / 'latency_stacked.png'}")
    print(f"[Saved] {out_dir / 'cut_points.png'}")


def dynamic_with_data(solution_dir: Path):
    # ========= 1) 找数据 ========
    npz_path = solution_dir / "solution.npz"
    json_path = solution_dir / "config.json"
    if not json_path.exists() or not npz_path.exists():
        print(f"Error: 路径 {solution_dir} 下缺少必要文件")
        return
    # ========= 2) 加载数据 ========
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    paras = Paras.from_dict(config.get("System_Parameters", {}))
    data = np.load(npz_path, allow_pickle=True)
    X, Y, F_e, F_c = data['X'], data['Y'], data['F_e'], data['F_c']
    # ========= 3) 画图 ========
    plot_user_dynamic(X, Y, F_e, F_c, paras)


def dynamic_without_data(n, F_u, H_u):
    # ========= 1) 构造自定义参数并运行 ========
    custom_paras_dict = {
        "n": n,
        "F_u": F_u,
        "H_u": H_u,
    }
    best_val, best_sol, history, paras = run_dsci_experiment(
        custom_paras_dict=custom_paras_dict,
        save_log=True
    )
    X, Y, F_e, F_c = best_sol
    # ========= 2) 画图 ========
    plot_user_dynamic(X, Y, F_e, F_c, paras)


if __name__ == "__main__":
    # 没有解的时候：
    n = 10
    F_u = np.array([0.1] * 3 + [2.0] * 4 + [8.0] * 3, dtype=np.float32)
    H_u = np.array([2.0] * 10, dtype=np.float32)
    dynamic_without_data(n, F_u, H_u)

    # 有解的时候：
    # dynamic_with_data(Path("D:\Coding\Python\DSCI\Result\PPO\PPO_20260123_150230"))