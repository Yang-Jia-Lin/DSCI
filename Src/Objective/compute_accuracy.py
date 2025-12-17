import numpy as np


def _get_acc(Y_ij, j, exit_rates):
    closest_idx = round(Y_ij * 100)

    # 确保整数在 0 到 100 之间
    if closest_idx < 0:
        closest_idx = 0
    elif closest_idx > 100:
        closest_idx = 100

    return exit_rates[closest_idx, j] / 100

def compute_expected_accuracy(Y, P, paras):
    n, m = Y.shape
    acc = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if j in paras.E:
                acc[i, j] = _get_acc(Y[i, j], j, paras.accs)
                # print(f"第{i}行，第{j}列，阈值为{Y[i,j]}，精度为{acc[i, j]}，退出概率为{P[i,j]}")
        acc[i,m-1]=0.8651
    accuracy = acc * P
    return np.sum(accuracy, axis=1)