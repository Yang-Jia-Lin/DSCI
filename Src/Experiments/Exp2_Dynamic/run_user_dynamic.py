"""
Src/Exp2_Dynamic/run_user_dynamic.py
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from Src.paras import Paras, RESULT_DYNAMIC_PATH
from Src.Optimizer.PPO.run_PPO import run_dsci_experiment
from Src.Experiments.Exp2_Dynamic.plot_decision import plot_X, plot_Y
from Src.Experiments.Exp2_Dynamic.plot_latency_stacked import plot_latency_stacked
from Src.Objective.compute_latency import compute_5_latency
from Src.Objective.compute_P import compute_layer_exit_probs


def plot_user_dynamic(X, Y, F_e, F_c, paras):
    """
    绘制一次决策中多个用户的 时延推叠图 X决策图 Y决策图
    """
    # ========= 1) 数据 ========
    P = compute_layer_exit_probs(Y, paras)
    T1, T2, T3, T4, T5 = compute_5_latency(X, P, F_e, F_c, paras)

    # ========= 2) 路径 ========
    out_dir = Path(RESULT_DYNAMIC_PATH) / f"UserHetero_{datetime.now().strftime('%m%d_%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    user_labels = [str(i + 1) for i in range(paras.n)]

    # ========= 3) 绘图 ========
    # 堆叠时延图
    plot_latency_stacked(user_labels,(T1, T2, T3, T4, T5), out_dir / "latency_stacked",)
    # X 决策热力图
    plot_X(X, paras.E, "Decisions", save_dir=out_dir)
    # Y 决策热力图
    plot_Y(Y, paras.E, "Decisions", save_dir=out_dir)


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
    # n = 18
    # F_u = np.array([0.1] * 6 + [1.0] * 6 + [8.0] * 6, dtype=np.float32)
    # H_u = np.array([2.0] * 18, dtype=np.float32)
    # dynamic_without_data(n, F_u, H_u)

    # 有解的时候：
    dynamic_with_data(Path("D:\Coding\Python\DSCI\Result\Optimize\PPO\PPO_20260129_025840"))