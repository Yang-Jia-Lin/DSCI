from __future__ import annotations
from typing import List
import numpy as np

from objective.compute_P import compute_layer_exit_probs
from objective.compute_accuracy import compute_expected_accuracy
from objective.compute_latency import compute_total_latency
from objective.objective import objective
from optimizer.SAC_d import SACDAgent
from utils.compute_paras import compute_iota_kappa, allocate_resources


def optimize_RL(paras, *,
                K_rl=50, outer_max_iter=50,
                epsilon=1e-3, batch_size=64,
                rl_agent_class=SACDAgent):
    # Initialize
    n, m = paras.n, paras.m
    agent = rl_agent_class(n_layers=m, early_layers=paras.E)

    f_e = np.full(n, paras.f_e_max / n)
    f_c = np.full(n, paras.f_c_max / n)

    best_val = -np.inf
    X_best = np.zeros((n, m), int)
    Y_best = np.zeros((n, m))
    f_e_best, f_c_best = f_e.copy(), f_c.copy()

    history: List[np.floating] = []
    U_prev = -np.inf
    f_e_prev = f_e.copy()
    f_c_prev = f_c.copy()

    # 2-Stage Optimize Loops
    for outer in range(outer_max_iter):
        print(f"\n--- Outer {outer} ---")

        # ===== Inner RL training =====
        for _ in range(K_rl):
            X_ep = np.zeros((n, m), int)
            Y_ep = np.zeros((n, m))
            phi_prev = 0.0
            # Calculate per user as t
            for t in range(n):
                ones = X_ep[t].sum()
                state = np.array([t / n, ones / 2.0], dtype=np.float32)

                # 构造掩码
                x_mask = np.ones(m, bool)
                x_mask[:] = ones < 2
                y_low = np.ones(m)  # 默认都只能 ≥1
                y_high = np.ones(m)  # 默认都只能 ≤1
                # 对早退层开放 [0,1]
                y_low[paras.E] = 0.0
                y_high[paras.E] = 1.0
                mask = {
                    "x_mask": x_mask,
                    "y_low": y_low,
                    "y_high": y_high,
                }

                x_row, y_row = agent.select_action(state, mask)
                X_ep[t], Y_ep[t] = x_row, y_row
                X_part = X_ep[: t + 1]
                Y_part = Y_ep[: t + 1]
                P_part = compute_layer_exit_probs(Y_part, paras)
                latency = compute_total_latency(X_part, P_part, f_e[: t + 1], f_c[: t + 1], paras).sum()
                acc = compute_expected_accuracy(Y_part, P_part, paras).sum()
                obj = objective(X_part, P_part, f_e[: t + 1], f_c[: t + 1], paras)
                phi_t = obj  # 潜在函数
                agent.store_transition(state, (x_row, y_row), phi_t - phi_prev, False)
                phi_prev = phi_t
            U_ep = objective(X_ep, Y_ep, f_e, f_c, paras)
            agent.store_transition(None, None, U_ep, True)
            agent.update(batch_size)
        X_cur, Y_cur = agent.eval_policy(n)  # 用当前策略评估

        # ===== Outer Opt =====
        P = compute_layer_exit_probs(Y_cur, paras)
        iota, kappa = compute_iota_kappa(X_cur, paras.C, P)
        f_e, f_c = allocate_resources(iota, kappa, paras.f_e_max, paras.f_c_max)

        # Evaluation
        U_cur = objective(X_cur, Y_cur, f_e, f_c, paras)
        history.append(U_cur)
        # 只有在 U_cur 上升时才更新，下降时保持原有结果
        if U_cur > U_prev:
            history.append(U_cur)
            f_e_prev, f_c_prev = f_e.copy(), f_c.copy()  # 记录当前的资源分配
        else:
            history.append(U_prev)
            f_e, f_c = f_e_prev.copy(), f_c_prev.copy()  # 恢复之前的资源分配
        P_cur = compute_layer_exit_probs(Y_cur, paras)
        latency_vec = compute_total_latency(X_cur, P_cur, f_e, f_c, paras)
        acc_vec = compute_expected_accuracy(Y_cur, P_cur, paras)
        print(f"latency: {sum(latency_vec)}")
        print(f"acc: {sum(acc_vec)}")
        print(f"Utility = {U_cur:.6f}")

        # Update best if U_cur is better
        if U_cur > best_val:
            best_val = U_cur
            X_best, Y_best = X_cur.copy(), Y_cur.copy()
            f_e_best, f_c_best = f_e.copy(), f_c.copy()

        # Update previous values
        U_prev = U_cur

    return best_val, (X_best, Y_best, f_e_best, f_c_best), history

