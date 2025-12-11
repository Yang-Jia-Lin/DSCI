# functions/compute_P.py

import numpy as np

# one user in one layer (p_ij)
def _get_independent_prob(Y_ij, j, exit_rates):
    closest_idx = round(Y_ij * 100)

    # 确保整数在 0 到 100 之间
    if closest_idx < 0:
        closest_idx = 0
    elif closest_idx > 100:
        closest_idx = 100

    return exit_rates[closest_idx, j] / 100




def compute_layer_exit_probs(Y, paras):
    n, m = Y.shape
    p = np.zeros((n, m))
    P = np.zeros((n, m))

    for i in range(n):
        # 1) 计算独立退出概率
        for j in range(m):
            if j in paras.E:
                p[i, j] = _get_independent_prob(Y[i, j], j, paras.rates)

        # 2) 计算组合退出概率
        for j in range(m):
            if j == 0:
                P[i, j] = p[i, j]
            else:
                P[i, j] = p[i, j] * np.prod(1 - p[i, :j])

        # 3) 整体归一化
        total = P[i].sum()
        if total > 0:
            P[i] = P[i] / total
        else:
            # 全零时，按业务给个默认分配，比如只退出最后一层
            P[i, -1] = 1

    return P
