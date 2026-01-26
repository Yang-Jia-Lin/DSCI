"""
Src/Dynamic/run_dynamic.py
"""
import json
import numpy as np
from pathlib import Path
from Src.paras import Paras
from Src.Optimizer.PPO.run_PPO import run_dsci_experiment
from Src.Dynamic.plot_dynamic import plot_user_dynamic


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