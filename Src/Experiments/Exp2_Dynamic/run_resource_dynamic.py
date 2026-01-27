"""
Src/Experiments/Exp2_Dynamic/run_resource_dynamic.py
"""
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from Src.Experiments.Exp2_Dynamic.plot_decision import plot_resource_trend
from Src.Optimizer.PPO.run_PPO import run_dsci_experiment
from Src.Utils.parsing_data import split_points_matrix
from Src.paras import RESULT_DYNAMIC_PATH


def resource_dynamic(resume_path: Path = None):
    """
    resume_path: 如果传入已有的文件夹路径，尝试从上次中断的地方继续
    """
    # 1. 实验目录管理
    if resume_path and resume_path.exists():
        exp_dir = resume_path
        print(f"Resuming experiment from: {exp_dir}")
    else:
        timestamp = datetime.now().strftime('%m%d_%H%M')
        exp_dir = Path(RESULT_DYNAMIC_PATH) / f"Resource_Hetero_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"New experiment started: {exp_dir}")
    csv_path = exp_dir / "res_dynamic_data.csv"

    # 2. 检查已有的进度
    completed_f_vals = []
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            completed_f_vals = df_existing["F_u"].tolist()
            print(f"Found existing data. {len(completed_f_vals)} steps completed.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV, starting fresh. Error: {e}")

    # 3. 实验配置
    num_users = 20
    f_u_range = np.arange(0.5, 8.5, 1)
    h_u_val = np.array([2.0] * num_users, dtype=np.float32)
    print("-" * 50)

    # 4. 遍历资源
    for i, f_val in enumerate(f_u_range):
        # --- 断点检测 ---
        if any(np.isclose(f_val, c_val) for c_val in completed_f_vals):
            print(f"[{i + 1}/{len(f_u_range)}] Skipping F_u = {f_val} GHz (Already exists).")
            continue
        print(f"[{i + 1}/{len(f_u_range)}] Running for User Frequency: {f_val} GHz...")

        custom_paras = {"n": num_users, "F_u": [f_val] * num_users, "H_u": h_u_val}
        try:
            # --- PPO优化 ---
            best_val, best_sol, history, paras = run_dsci_experiment(
                custom_paras_dict=custom_paras, save_log=True
            )

            # --- 解析决策 X ---
            X, Y, F_e, F_c = best_sol
            cut_points = split_points_matrix(np.array(X))
            clean_cuts = cut_points.astype(float)
            clean_cuts[clean_cuts[:, 0] == -1, 0] = paras.m
            avg_end_edge = np.mean(clean_cuts[:, 0])
            avg_edge_cloud = np.mean(clean_cuts[:, 1])

            # --- 构造当前行数据 --
            new_row = {
                "F_u": f_val,
                "avg_end_edge": avg_end_edge,
                "avg_edge_cloud": avg_edge_cloud,
                "total_utility": best_val
            }

            # --- 即时增量写入CSV ---
            df_new_row = pd.DataFrame([new_row])
            # 如果文件不存在，写入表头；如果存在，则追加且不写表头
            df_new_row.to_csv(csv_path, mode='a', index=False, header=not csv_path.exists())
            print(f"   Success! End-Edge: {avg_end_edge:.1f}, Utility: {best_val:.2f} (Saved to CSV)")

        except KeyboardInterrupt:
            print("\nExperiment interrupted by user. Progress is saved.")
            return csv_path, exp_dir
        except Exception as e:
            print(f"   Error at F_u = {f_val}: {e}. Skipping to next...")
            continue

    print("-" * 50 + f"\n[All Finished] Results in: {exp_dir}")

    # 绘图
    plot_resource_trend(
        csv_path=csv_path,
        save_dir=exp_dir,
        total_layers=paras.m
    )
    return csv_path, exp_dir


if __name__ == "__main__":
    # 从某文件夹继续
    # resource_dynamic(resume_path=Path("D:\Coding\Python\DSCI\Result\Exp2_Dynamic\Resource_Hetero_0520_1400"))
    # 从头运行
    csv_file, plot_dir = resource_dynamic()