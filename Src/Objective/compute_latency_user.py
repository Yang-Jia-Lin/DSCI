"""
计算单一任务时延
Src/Objective/compute_latency_user.py
"""

import numpy as np


def _compute_end_to_edge_delay(d_i, h_i, B_e, G, delta):
    R_i = (B_e * 1e6) * np.log2(1 + (h_i * G) / delta)
    return d_i / R_i


def _compute_edge_to_cloud_delay(d_e2c, B_c):
    return d_e2c / (B_c * 1e6)


def _compute_local_computation_delay(cut_points_i, P_i, C_i, f_u):
    cut0 = int(cut_points_i[0])
    m = len(C_i)

    if cut0 <= 0:
        return 0.0
    if cut0 > m:
        cut0 = m

    f_u = float(f_u) * 1e9
    if f_u <= 0:
        return float("inf")

    T_expected = 0.0

    # (1) local 退出：j in [0, cut0)
    for j in range(0, cut0):
        T_expected += float(P_i[j]) * (float(np.sum(C_i[: j + 1])) / f_u)

    # (2) 后面退出：本地必须算满到 cut0-1
    prob_reach_edge = float(np.sum(P_i[cut0:])) if cut0 < m else 0.0
    if prob_reach_edge > 0:
        T_expected += prob_reach_edge * (float(np.sum(C_i[:cut0])) / f_u)

    return float(T_expected)


def _compute_edge_computation_delay(cut_points_i, P_i, C, f_e):
    cut0 = int(cut_points_i[0])
    cut1 = int(cut_points_i[1])
    m = len(C)

    if cut0 < 0:
        cut0 = 0
    if cut0 > m:
        cut0 = m

    effective_cut1 = m if cut1 == -1 else cut1
    if effective_cut1 < 0:
        effective_cut1 = 0
    if effective_cut1 > m:
        effective_cut1 = m

    if effective_cut1 <= cut0:
        return 0.0

    f_e = float(f_e) * 1e9
    if f_e <= 0:
        return float("inf")

    T_expected = 0.0

    # (1) edge 段退出：j in [cut0, effective_cut1)
    for j in range(cut0, effective_cut1):
        T_expected += float(P_i[j]) * (float(np.sum(C[cut0: j + 1])) / f_e)

    # (2) 进入 cloud：edge 必须算满 [cut0, effective_cut1)
    if effective_cut1 < m:
        prob_reach_cloud = float(np.sum(P_i[effective_cut1:]))
        if prob_reach_cloud > 0:
            T_expected += prob_reach_cloud * (float(np.sum(C[cut0: effective_cut1])) / f_e)

    return float(T_expected)


def _compute_cloud_computation_delay(cut_points_i, P_i, C, f_c):
    cut1 = int(cut_points_i[1])
    m = len(C)

    if cut1 == -1 or cut1 >= m:
        return 0.0
    if cut1 < 0:
        cut1 = 0

    f_c = float(f_c) * 1e9
    if f_c <= 0:
        return float("inf")

    T_expected = 0.0
    for j in range(cut1, m):
        T_expected += float(P_i[j]) * (float(np.sum(C[cut1: j + 1])) / f_c)

    return float(T_expected)


def compute_user_latency(
    u: int,
    cut0: int,
    cut1: int,
    P_row: np.ndarray,
    F_e_u: float,
    F_c_u: float,
    paras,
) -> float:
    """
    严格等价于 compute_total_latency() 里对单个用户 i 的那一段计算。
    只计算用户 u 的期望总时延 T[u]。

    参数:
      - cut0, cut1: 由 compute_exit_points 得到的切分点（cut1 可能为 -1）
      - P_row: P[u]，长度 m
      - F_e_u, F_c_u: 分配给该用户的 edge/cloud 频率 (GHz，与你的 compute_total_latency 一致)
    """
    C = np.asarray(paras.C, dtype=np.float64)
    D = np.asarray(paras.D, dtype=np.float64)
    H = np.asarray(paras.H_u, dtype=np.float64).reshape(-1)
    F_u = np.asarray(paras.F_u, dtype=np.float64).reshape(-1)

    b_e = float(paras.b_e)
    b_c = float(paras.b_c)
    G = float(paras.G)
    delta = float(paras.delta)

    m = len(C)

    cut0 = int(cut0)
    cut1 = int(cut1)

    P_i = np.asarray(P_row, dtype=np.float64).reshape(-1)

    f_e = float(F_e_u)
    f_c = float(F_c_u)
    f_u = float(F_u[u])
    h_i = float(H[u])

    T = 0.0

    # ---- Local computation ----
    if cut0 > 0:
        T += _compute_local_computation_delay((cut0, cut1), P_i, C, f_u)

    # 进入 edge 的概率（退出层 >= cut0）
    prob_reach_edge = float(np.sum(P_i[cut0:])) if 0 <= cut0 < m else 0.0

    # ---- U->E transmission & Edge computation ----
    if 0 <= cut0 < m and prob_reach_edge > 0:
        T += prob_reach_edge * _compute_end_to_edge_delay(float(D[cut0]), h_i, b_e, G, delta)
        T += _compute_edge_computation_delay((cut0, cut1), P_i, C, f_e)

    # 进入 cloud 的概率（退出层 >= cut1），仅当 cut1 有效且存在 cloud 段
    prob_reach_cloud = float(np.sum(P_i[cut1:])) if 0 <= cut1 < m else 0.0

    # ---- E->C transmission & Cloud computation ----
    if 0 <= cut1 < m and cut1 != -1 and prob_reach_cloud > 0:
        T += prob_reach_cloud * _compute_edge_to_cloud_delay(float(D[cut1]), b_c)
        T += _compute_cloud_computation_delay((cut0, cut1), P_i, C, f_c)

    return float(T)
