import numpy as np
from utils.parsing_data import split_points_matrix


def compute_iota_kappa(X, compute_sizes, exit_prob):
    """计算拉格朗日参数 iota 和 kappa"""
    n, m = X.shape
    c = np.asarray(compute_sizes)
    iota = np.zeros(n)
    kappa = np.zeros(n)
    split_pts = split_points_matrix(X)
    for i in range(n):
        p1, p2 = split_pts[i]
        for j in range(p1 + 1, p2 + 1):
            iota[i]  += exit_prob[i, j] * c[p1 + 1 : j + 1].sum()
        for j in range(p2 + 1, m):
            kappa[i] += exit_prob[i, j] * c[p2 + 1 : j + 1].sum()
    return iota, kappa


def allocate_resources(iota, kappa, f_e_max, f_c_max):
    """计算凸优化后的资源分配"""
    sqrt_i, sqrt_k = np.sqrt(iota + 1e-12), np.sqrt(kappa + 1e-12)
    f_e = f_e_max * sqrt_i / max(sqrt_i.sum(), 1e-12)
    f_c = f_c_max * sqrt_k / max(sqrt_k.sum(), 1e-12)
    return f_e, f_c