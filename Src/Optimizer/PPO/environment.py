"""
Src/Optimizer/PPO/environment.py
"""

from Src.Objective.objective import objective
import numpy as np


# 约束 X
def project_X(X):
    """
    保证X每行最多2个1，其余置0
    """
    for i in range(X.shape[0]):
        if np.sum(X[i]) > 2:
            # 直接随机保留2个
            ones_idx = np.where(X[i]==1)[0]
            keep_idx = np.random.choice(ones_idx, size=2, replace=False)
            X[i] = 0
            X[i][keep_idx] = 1
    return X


# 约束 Y
def clip_Y(Y, E):
    """
    对Y clip到[0,1]，并固定不可早退层(j not in E)为1
    """
    Y = np.clip(Y, 0, 1)
    all_indices = np.arange(Y.shape[1])
    fixed_indices = np.setdiff1d(all_indices, E)
    Y[:, fixed_indices] = 1.0
    return Y


# 计算奖励
def compute_reward(X,Y,F_e,F_c,paras):
    return objective(X,Y,F_e,F_c,paras)


# 随机初始化
def init_feasible_XY(paras):
    n, m = paras.n, paras.m
    X = np.random.randint(0, 2, (n, m))
    X = project_X(X)
    Y = np.random.uniform(0, 1, (n, m))
    Y = clip_Y(Y, paras.E)
    return X, Y