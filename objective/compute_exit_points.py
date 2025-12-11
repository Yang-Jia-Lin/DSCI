# functions/compute_exit_points.py

import numpy as np


# one user exit points
def _compute_one_exit_points(X_i):
    ones = np.where(X_i == 1)[0]

    if len(ones) == 0:
        L = len(X_i)
        return L-1, L-1

    # 第一个分割点
    p_i1 = ones[0]
    # 第二个分割点（如果有的话）
    tail = X_i[p_i1 + 1:]
    next_ones = np.where(tail == 1)[0]
    if len(next_ones) > 0:
        p_i2 = next_ones[0] + p_i1 + 1
    else:
        p_i2 = -1

    return p_i1, p_i2



# all users exit points
def compute_exit_points(X, paras):
    n = X.shape[0]
    cut_points = np.zeros((n, 2), dtype=int)
    for i in range(n):
        cut_points[i] = _compute_one_exit_points(X[i])
    return cut_points


