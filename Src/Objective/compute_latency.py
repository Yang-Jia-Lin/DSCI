# function/compute_latency.py

from Src.paras import *
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_exit_points import compute_exit_points
from Src.Utils.parsing_data import parsing_rate_and_acc


def _compute_end_to_edge_delay(d_i, h_i, B_e, G, delta):
    R_i = (B_e * 1e6) * np.log2(1 + (h_i * G) / delta)
    return d_i / R_i


def _compute_edge_to_cloud_delay(d_e2c, B_c):
    return d_e2c / (B_c * 1e6)


def _compute_local_computation_delay(cut_points_i, P_i, C_i, f_u):
    """
    Local Range: [0, cut0)
    例如 cut0=0，则 range(0,0) 为空，时延为0。
    """
    cut0 = int(cut_points_i[0])

    # 如果 cut0=0，说明端设备不计算任何层，直接卸载
    if cut0 <= 0:
        return 0.0

    # 如果分配的算力资源太小，说明计算时间很长，直接返回inf
    f_u = float(f_u) * 1e9
    if f_u <= 0:
        return float("inf")

    # 计算范围：[0, cut0)
    T_expected = 0.0
    for j in range(0, cut0):
        T_expected += P_i[j] * (sum(C_i[:j + 1]) / f_u)
    return float(T_expected)


def _compute_edge_computation_delay(cut_points_i, P_i, C, f_e):
    """
    Edge Range: [cut0, cut1)
    注意：需处理 cut1=-1 的情况，代表 Edge 负责到底
    """
    cut0 = int(cut_points_i[0])
    cut1 = int(cut_points_i[1])
    m = len(C)

    # 预处理 cut1：如果是 -1，表示没有切给云，Edge 负责到最后 (m)
    effective_cut1 = m if cut1 == -1 else cut1

    # 如果 range 为空 (如 cut0=2, cut1=2)，则 Edge 不计算
    if effective_cut1 <= cut0:
        return 0.0

    # 如果分配的算力资源太小，说明计算时间很长，直接返回inf
    f_e = float(f_e) * 1e9
    if f_e <= 0:
        return float("inf")

    # 计算范围：[cut0, effective_cut1)
    T_expected = 0.0
    for j in range(cut0, effective_cut1):
        T_expected += P_i[j] * (sum(C[cut0: j + 1]) / f_e)
    return float(T_expected)


def _compute_cloud_computation_delay(cut_points_i, P_i, C, f_c):
    """
    Cloud Range: [cut1, m)
    """
    cut1 = int(cut_points_i[1])
    m = len(C)

    # 如果 cut1=-1 (Edge负责到底) 或 cut1=m (Cloud无任务)，直接返回
    if cut1 == -1 or cut1 >= m:
        return 0.0

    f_c = float(f_c) * 1e9
    if f_c <= 0: return float("inf")

    # 计算范围：[cut1, m)
    T_expected = 0.0
    for j in range(cut1, m):
        T_expected += P_i[j] * (sum(C[cut1: j + 1]) / f_c)
    return float(T_expected)


def compute_total_latency(X, P, F_e, F_c, paras):
    n = X.shape[0]
    m = X.shape[1]
    C = paras.C
    D = paras.D
    H = paras.H_u
    F_u = paras.F_u
    b_e = paras.b_e
    b_c = paras.b_c
    G = paras.G
    delta = paras.delta

    T = np.zeros(n, dtype=np.float64)
    cut_points = compute_exit_points(X, paras)

    for i in range(n):
        T[i] = 0
        cut0 = int(cut_points[i][0])
        cut1 = int(cut_points[i][1])

        P_i = P[i]
        f_e = float(np.asarray(F_e).reshape(-1)[i])
        f_c = float(np.asarray(F_c).reshape(-1)[i])
        f_u = float(np.asarray(F_u).reshape(-1)[i])
        h_i = float(np.asarray(H).reshape(-1)[i])

        # ---- Local computation ----
        if cut0 > 0:
            T[i] += _compute_local_computation_delay((cut0, cut1), P_i, C, f_u)

        # ---- U->E transmission (only if cut0 exists) ----
        if cut0 < m:
            T[i] += _compute_end_to_edge_delay(float(D[cut0]), h_i, b_e, G, delta)
            # ---- Edge computation ----
            T[i] += _compute_edge_computation_delay((cut0, cut1), P_i, C, f_e)

        # ---- E->C transmission (only if cut1 exists) ----
        if 0 < cut1 < m:
            d_i_2 = float(D[cut1])
            T[i] += _compute_edge_to_cloud_delay(d_i_2, b_c)
            # ---- Cloud computation ----
            T[i] += _compute_cloud_computation_delay((cut0, cut1), P_i, C, f_c)

    return T


# ==========================================
# Test Block for Latency
# ==========================================
if __name__ == "__main__":
    print(">>> 初始化参数...")
    paras = Paras(
        n=NUM_USERS, m=NUM_LAYERS, E=EARLY_EXIT_LAYERS, D=DATA_SIZE_LAYERS, C=COMPUTE_SIZE_LAYERS,
        F_u=USER_FREQs, f_e_max=EDGE_MAX_FREQ, f_c_max=CLOUD_MAX_FREQ, H_u=CHANNEL_GAINS_USERS,
        b_e=BANDWIDTH_EDGE, b_c=BANDWIDTH_CLOUD, G=BASE_STATION_POWER, delta=NOISE_POWER,
        alpha=1, beta=20
    )
    paras.rates, paras.accs = parsing_rate_and_acc(paras)
    n, m = paras.n, paras.m

    # 2. 准备输入数据
    X = np.zeros((n, m))
    X[0][0] = 1
    X[0][50] = 1

    Y = np.ones((n, m))
    Y[0, 57] = 0.9
    Y[0, 103] = 0.8

    F_e = np.ones((n, 1)) * (paras.f_e_max / n)
    F_c = np.ones((n, 1)) * (paras.f_c_max / n)

    P = compute_layer_exit_probs(Y, paras)
    cut_points = compute_exit_points(X, paras)
    c0 = int(cut_points[0][0])
    c1 = int(cut_points[0][1])


    # 3. 手动拆解时延计算 (使用刚算出来的 c0, c1)
    print(f"\n{'=' * 20} User 0 Latency Breakdown {'=' * 20}")
    print(f"cut point is ({c0}, {c1})")
    P_i = P[0]
    f_e = F_e[0].item()
    f_c = F_c[0].item()
    f_u = float(paras.F_u[0])
    h_i = float(paras.H_u[0])
    cut_tuple = (c0, c1)

    # A. Local
    t_local = _compute_local_computation_delay(cut_tuple, P_i, paras.C, f_u)
    if c0 > 0:
        print(f"1. Local Comp Layers [0, {c0}): \t{t_local:.12f} s")
    else:
        print(f"1. Local Comp (None): \t0 s")

    # B. Trans U->E
    if c0 < m:
        data_u2e = float(paras.D[c0])
        t_trans_1 = _compute_end_to_edge_delay(data_u2e, h_i, paras.b_e, paras.G, paras.delta)
        print(f"2. Trans U->E (Data D[{c0}]):   \t{t_trans_1:.12f} s")
    else:
        t_trans_1 = 0.0
        print(f"2. Trans U->E (None):   \t0 s")

    # C. Edge Comp
    if c0 < m:
        t_edge = _compute_edge_computation_delay(cut_tuple, P_i, paras.C, f_e)
        print(f"3. Edge Comp Layers [{c0}, {c1}): \t{t_edge:.12f} s")
    else:
        t_edge = 0.0
        print(f"3. Edge Comp Layers (None): \t0 s")

    # D. Trans E->C
    if 0 < c1 < m:
        data_e2c = float(paras.D[c1])
        t_trans_2 = _compute_edge_to_cloud_delay(data_e2c, paras.b_c)
        print(f"4. Trans E->C (Data D[{c1}]):   \t{t_trans_2:.12f} s")
    else:
        t_trans_2 = 0.0
        print(f"4. Trans E->C (None):   \t0 s")

    # E. Cloud Comp
    if 0 < c1 < m:
        t_cloud = _compute_cloud_computation_delay(cut_tuple, P_i, paras.C, f_c)
        print(f"5. Cloud Comp Layers [{c1}, m): \t{t_cloud:.12f} s")
    else:
        t_cloud = 0.0
        print(f"5. Cloud Comp Layers (None): \t0 s")

    # F. Sum
    manual_total = t_local + t_trans_1 + t_edge + t_trans_2 + t_cloud
    print("-" * 60)
    print(f"Manual Total Sum: \t\t\t{manual_total:.6f} s")


    # 6. 调用主函数 compute_total_latency 进行验证
    print("\n>>> 调用 compute_total_latency 函数...")
    T_vec = compute_total_latency(X, P, F_e, F_c, paras)
    func_total = T_vec[0]
    print(f"Function Result: \t\t\t{func_total:.6f} s")


    # 7. 最终校验
    diff = abs(manual_total - func_total)
    if diff < 1e-9:
        print("\n✅ 测试通过：手动拆解计算结果与主函数结果一致。")
    else:
        print(f"\n❌ 测试失败：误差为 {diff}。")