"""
Src/Experiments/Exp1_Baseline/run_base_baseline.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from Src.Experiments.Exp1_Baseline.plot_baseline import plot_bubble_chart, plot_utility_bar
from Src.paras import Paras
from Src.Objective.objective import objective
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_latency import compute_total_latency
from Src.Objective.compute_accuracy import compute_expected_accuracy
from Src.Utils.log_function import load_and_analyze_results


def evaluate_strategy(X, Y, F_e, F_c, paras, name: str = "Strategy") -> Dict[str, float]:
    P = compute_layer_exit_probs(Y, paras)
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    acc_vec = compute_expected_accuracy(Y, P, paras)
    obj_val = objective(X, Y, F_e, F_c, paras)
    weighted_latency = paras.beta * np.sum(latency_vec)
    weighted_acc = paras.alpha * np.sum(acc_vec)
    return {
        "name": name,
        "latency": weighted_latency,
        "accuracy": weighted_acc,
        "objective": obj_val
    }


def get_baseline_decisions(mode: str, paras, X_opt, Y_opt, F_e_opt, F_c_opt) -> Tuple:
    n, m = paras.n, paras.m
    X = np.zeros((n, m))
    Y = np.ones((n, m))
    Y[:, m - 1] = 0

    if mode == "Device":
        # 仅在终端：X全0，Y全1，资源极低
        return X, Y, F_e_opt, F_c_opt

    elif mode == "Edge":
        # 仅在边端：在第一层切分
        X[:, 0] = 1
        return X, Y, F_e_opt, F_c_opt

    elif mode == "Cloud":
        # 仅在云端：在第一、二层切分
        X[:, 0] = 1
        X[:, 1] = 1
        return X, Y, F_e_opt, F_c_opt

    elif mode == "Collaborative":
        # 端边云协同，无早退
        return X_opt, Y, F_e_opt, F_c_opt

    elif mode == "Edge+EE":
        # 边端 + 早退
        X[:, 0] = 1
        return X, Y_opt, F_e_opt, F_c_opt

    elif mode == "Collaborative+EE":
        # 端边云协同+早退（DSCI）
        return X_opt, Y_opt, F_e_opt, F_c_opt

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # 1. 加载数据
    PPO_path = Path(r"D:\Coding\Python\DSCI\Result\Optimize\PPO\PPO_20260127_153244")
    X_opt, Y_opt, F_e_opt, F_c_opt, _, paras = load_and_analyze_results(exp_dir=PPO_path, analysis=False)

    # 2. 定义 Baseline 列表
    baseline_modes = [
        ("仅在终端", "Device"),
        ("仅在边端", "Edge"),
        ("仅在云端", "Cloud"),
        ("仅协同", "Collaborative"),
        ("边端 + 早退", "Edge+EE"),
        ("协同 + 早退", "Collaborative+EE")
    ]

    # 3. 循环计算并打印结果
    results = []
    print(f"{'策略名称':<15} | {'Latency':<10} | {'Accuracy':<10} | {'Objective':<10}")
    print("-" * 60)
    for display_name, mode in baseline_modes:
        X, Y, Fe, Fc = get_baseline_decisions(mode, paras, X_opt, Y_opt, F_e_opt, F_c_opt)
        res = evaluate_strategy(X, Y, Fe, Fc, paras, name=display_name)
        results.append(res)
        print(f"{res['name']:<15} | {res['latency']:<10.4f} | {res['accuracy']:<10.4f} | {res['objective']:<10.4f}")

    # 4. 绘图
    df = pd.DataFrame(results)
    df['latency_ms'] = (df['latency'] / paras.beta)
    df['accuracy'] = (df['accuracy'] / paras.alpha)
    plot_bubble_chart(df)
    plot_utility_bar(df)