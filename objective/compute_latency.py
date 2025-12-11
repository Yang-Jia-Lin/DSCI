# function/compute_latency.py

import numpy as np

from objective.compute_exit_points import compute_exit_points


# 通信时延 端-边
def _compute_end_to_edge_delay(d_i, h_i, B_e, G, delta):
    R_i = (B_e * 1e6) * np.log2(1 + (h_i * G) / delta)
    T_i_trans = d_i / R_i
    return T_i_trans


# 通信时延 边-云
def _compute_edge_to_cloud_delay(d_e2c, B_c):
    T_e2c = d_e2c / (B_c * 1e6)
    return T_e2c


# 计算时延 本地
def _compute_local_computation_delay(cut_points_i, P_i, C_i, f_u):
    T_l_i = 0
    f_u *= 1e9
    for j in range(cut_points_i[0]+1):
        T_l_i += P_i[j] * sum(C_i[:j]) / f_u
    return T_l_i


# 计算时延 边缘
def _compute_edge_computation_delay(cut_points_i, P_i, C, f_e):
    T_e_i = 0
    f_e *= 1e9
    for j in range(cut_points_i[0], cut_points_i[1]):
        # print(f"第{j}层，计算量{sum(C[cut_points_i[0]:j])}，计算资源{f_e}")
        T_e_i += P_i[j] * sum(C[cut_points_i[0]:j]) / f_e
    return T_e_i


# 计算时延 云
def _compute_cloud_computation_delay(cut_points_i, P_i, C, f_c):
    T_c_i = 0
    f_c *= 1e9
    for j in range(cut_points_i[1], len(C)):
        T_c_i += P_i[j] * sum(C[cut_points_i[1]:j]) / f_c
    return T_c_i


# 总时延
def compute_total_latency(X, P, F_e, F_c, paras):
    n = X.shape[0]
    C = paras.C
    D = paras.D
    H = paras.H_u
    F_u = paras.F_u
    b_e = paras.b_e
    b_c = paras.b_c
    G = paras.G
    delta = paras.delta

    T = np.zeros(n)
    cut_points = compute_exit_points(X, paras)

    for i in range(n):
        cut_points_i = cut_points[i]
        P_i = P[i]
        f_e = F_e[i]
        f_c = F_c[i]
        f_u = F_u[i]
        d_i_1 = D[cut_points_i[0]]
        d_i_2 = D[cut_points_i[1]]
        h_i = H[i]
        compute_user = _compute_local_computation_delay(cut_points_i, P_i, C, f_u)
        # print(f"compute_local_computation_delay is {compute_user}")
        transmit_u2e = _compute_end_to_edge_delay(d_i_1, h_i, b_e, G, delta)
        # print(f"compute_end_to_edge_delay is {transmit_u2e}")
        compute_edge = _compute_edge_computation_delay(cut_points_i, P_i, C, f_e)
        # print(f"compute_edge_computation_delay is {compute_edge}")
        transmit_e2c = _compute_edge_to_cloud_delay(d_i_2, b_c)
        # print(f"compute_edge_to_cloud_delay is {transmit_e2c}")
        compute_cloud = _compute_cloud_computation_delay(cut_points_i, P_i, C, f_c)
        # print(f"compute_cloud_computation_delay is {compute_cloud}\n")
        if cut_points_i[0] == -1:
            transmit_u2e = 0
        if cut_points_i[0] == -1:
            transmit_u2e = 0
        T[i] = compute_user + transmit_u2e + compute_edge + compute_cloud + transmit_e2c
    return T
