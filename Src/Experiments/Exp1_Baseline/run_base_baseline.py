from Src.paras import *
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Objective.objective import objective
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_latency import compute_total_latency
from Src.Objective.compute_accuracy import compute_expected_accuracy

paras = Paras(
    # Users
    n=NUM_USERS,
    # Model
    m=NUM_LAYERS,
    E=EARLY_EXIT_LAYERS,
    D=DATA_SIZE_LAYERS,
    C=COMPUTE_SIZE_LAYERS,
    # Resource
    F_u=USER_FREQs,
    f_e_max=EDGE_MAX_FREQ,
    f_c_max=CLOUD_MAX_FREQ,
    H_u=CHANNEL_GAINS_USERS,
    b_e = BANDWIDTH_EDGE,
    b_c = BANDWIDTH_CLOUD,
    G = BASE_STATION_POWER,
    delta = NOISE_POWER,
    # Weights
    alpha=0.6,
    beta=0.4
)
paras.rates, paras.accs = parsing_rate_and_acc(paras)
n = paras.n
m = paras.m


# ==== 仅在终端 ====
# X为全零矩阵，Y为全 1 矩阵
X = np.zeros((n, m))
Y = np.ones((n, m))
Y[:,m-1] = 0
F_e = np.zeros(n)+1e-6
F_c = np.zeros(n)+1e-6
P = compute_layer_exit_probs(Y, paras)
latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
print("仅在终端:")
print(f"latency:{paras.beta * sum(latency_vec)}")
acc_vec = compute_expected_accuracy(Y, P, paras)
print(f"acc:{paras.alpha * sum(acc_vec)}")
obj = objective(X, Y, F_e, F_c, paras)
print(f"obj:{obj}\n")

# ==== 仅在边端 ====
# X除了1列外为全零矩阵，Y为全 1 矩阵
X_0 = np.zeros((n, m))
X_0[:,0] = 1
Y_0 = np.ones((n, m))
Y_0[:,m-1] = 0
F_e_0 = np.ones(n)*EDGE_MAX_FREQ / n
F_c_0 = np.ones(n)*CLOUD_MAX_FREQ / n
P_0 = compute_layer_exit_probs(Y_0, paras)
latency_vec_0 = compute_total_latency(X_0, P_0, F_e_0, F_c_0, paras)
acc_vec_0 = compute_expected_accuracy(Y_0, P_0, paras)
obj_0 = objective(X_0, Y_0, F_e_0, F_c_0, paras)
print("仅在边端:")
print(f"latency:{paras.beta * sum(latency_vec_0)}")
print(f"acc:{paras.alpha * sum(acc_vec_0)}")
print(f"obj:{obj_0}\n")

# ==== 仅在云端 ====
# X除了12列外为全零矩阵，Y为全 1 矩阵
X_1 = np.zeros((n, m))
X_1[:,0] = 1
X_1[:,1] = 1
Y_1 = np.ones((n, m))
Y_1[:,m-1] = 0
F_e_1 = np.ones(n)*EDGE_MAX_FREQ / n
F_c_1 = np.ones(n)*CLOUD_MAX_FREQ / n
P_1 = compute_layer_exit_probs(Y_1, paras)
latency_vec_1 = compute_total_latency(X_1, P_1, F_e_1, F_c_1, paras)
acc_vec_1 = compute_expected_accuracy(Y_1, P_1, paras)
obj_1 = objective(X_1, Y_1, F_e_1, F_c_1, paras)
print("仅在云端:")
print(f"latency:{paras.beta * sum(latency_vec_1)}")
print(f"acc:{paras.alpha * sum(acc_vec_1)}")
print(f"obj:{obj_1}\n")

# ==== 协同 ====
X_2 = np.zeros((n, m))
X_2[:,57] = 1
X_2[:,103] = 1
Y_2 = np.ones((n, m))
# Y_c1[:,57] = 0.8
# Y_c1[:,103] = 0.5
Y_2[:,m-1] = 0
F_e_2 = np.ones(n)*EDGE_MAX_FREQ / n
F_c_2 = np.ones(n)*CLOUD_MAX_FREQ / n
P_2 = compute_layer_exit_probs(Y_2, paras)
latency_vec_2 = compute_total_latency(X_2, P_2, F_e_2, F_c_2, paras)
acc_vec_2 = compute_expected_accuracy(Y_2, P_2, paras)
obj_2 = objective(X_2, Y_2, F_e_2, F_c_2, paras)
print("协同:")
print(f"latency:{paras.beta * sum(latency_vec_2)}")
print(f"acc:{paras.alpha * sum(acc_vec_2)}")
print(f"obj:{obj_2}\n")

# ==== 早退 ====（在云端）
X_3 = np.zeros((n, m))
X_3[:,0] = 1
X_3[:,1] = 1
Y_3 = np.ones((n, m))
Y_3[:,57] = 0.8
Y_3[:,103] = 0.5
Y_3[:,m-1] = 0
F_e_3 = np.ones(n)*EDGE_MAX_FREQ / n
F_c_3 = np.ones(n)*CLOUD_MAX_FREQ / n
P_3 = compute_layer_exit_probs(Y_3, paras)
latency_vec_3 = compute_total_latency(X_3, P_3, F_e_3, F_c_3, paras)
acc_vec_3 = compute_expected_accuracy(Y_3, P_3, paras)
obj_3 = objective(X_3, Y_3, F_e_3, F_c_3, paras)
print("在云端+早退:")
print(f"latency:{paras.beta * sum(latency_vec_3)}")
print(f"acc:{paras.alpha * sum(acc_vec_3)}")
print(f"obj:{obj_3}\n")

# ==== 协同 + 早退 ====
X_4 = np.zeros((n, m))
X_4[:,57] = 1
X_4[:,103] = 1
Y_4 = np.ones((n, m))
Y_4[:,57] = 0.8
Y_4[:,103] = 0.5
Y_4[:,m-1] = 0
F_e_4 = np.ones(n)*EDGE_MAX_FREQ / n
F_c_4 = np.ones(n)*CLOUD_MAX_FREQ / n
P_4 = compute_layer_exit_probs(Y_4, paras)
latency_vec_4 = compute_total_latency(X_4, P_4, F_e_4, F_c_4, paras)
acc_vec_4 = compute_expected_accuracy(Y_4, P_4, paras)
obj_4 = objective(X_4, Y_4, F_e_4, F_c_4, paras)
print("协同+早退:")
print(f"latency:{paras.beta * sum(latency_vec_4)}")
print(f"acc:{paras.alpha * sum(acc_vec_4)}")
print(f"obj:{obj_4}\n")