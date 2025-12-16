from objective.compute_P import compute_layer_exit_probs
from objective.compute_latency import compute_total_latency
from paras import *
from utils.parsing_data import parsing_rate_and_acc
from objective.objective import objective
from utils.plot_results import plot_decisions, plot_convergence

from optimizer.GA import optimize_GA
from optimizer.RL import optimize_RL
from optimizer.PPO import optimize_PPO
from optimizer.BF import optimize_BF


# Parameters（大写向量(除G外)，小写标量）
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
    alpha=1,
    beta=20
)
paras.rates, paras.accs = parsing_rate_and_acc(paras)
n = paras.n
m = paras.m

# 变量
X = np.zeros([n,m])
Y  = np.ones([n,m])
F_e = np.ones((n, 1)) * paras.f_e_max / n
F_c = np.ones((n, 1)) * paras.f_c_max / n

# Objective
obj = objective(X, Y, F_e, F_c, paras)
print(f"objective is {obj}")

# Optimize GA
# GA_best_val, GA_best_sol, GA_history = optimize_GA(paras)
# GA_X_opt, GA_Y_opt, GA_F_e_opt, GA_F_c_opt = GA_best_sol
# plot_convergence(GA_history, data_name="GA_Convergence_history", save_dir=Path(RESULT_GA_PATH))
# plot_decisions(GA_X_opt, GA_Y_opt, data_name="GA最优结果示意图", save_dir=Path(RESULT_GA_PATH))

# Optimize RL
# RL_best_val, RL_best_sol, RL_history = optimize_RL(paras)
# RL_X_opt, RL_Y_opt, RL_F_e_opt, RL_F_c_opt = RL_best_sol
# print(RL_Y_opt[:,[57,103]])
# plot_convergence(RL_history, data_name="RL_Convergence_history", save_dir=Path(RESULT_RL_PATH))
# plot_decisions(RL_X_opt, RL_Y_opt, data_name="RL最优结果示意图", save_dir=Path(RESULT_RL_PATH))

# # Optimize PPO
# PPO_best_val, PPO_best_sol, PPO_history = optimize_PPO(paras)
# PPO_X_opt, PPO_Y_opt, PPO_F_e_opt, PPO_F_c_opt = PPO_best_sol
# print(PPO_Y_opt[:,[57,103]])
# plot_convergence(PPO_history, data_name="PPO_Convergence_history", save_dir=Path(RESULT_PPO_PATH))
# plot_decisions(PPO_X_opt, PPO_Y_opt, data_name="PPO最优结果示意图", save_dir=Path(RESULT_PPO_PATH))

# Optimize BF
# y_step=0.05 意味着每个阈值尝试 0, 0.05, 0.1 ... 1.0 (21个点)。
# 如果速度太慢，可以改为 0.1；如果想更精确，改为 0.01。
BF_best_val, BF_best_sol, BF_history = optimize_BF(paras, max_iter=5, y_step=0.1)
BF_X_opt, BF_Y_opt, BF_F_e_opt, BF_F_c_opt = BF_best_sol

print(f"BF Optimal Value: {BF_best_val}")
print("BF Latency:", np.sum(compute_total_latency(BF_X_opt, compute_layer_exit_probs(BF_Y_opt, paras), BF_F_e_opt, BF_F_c_opt, paras)))
# 验证阈值选择
# 假设 exit_indices 是 [57, 103]
print("BF Thresholds Sample:", BF_Y_opt[:, [57, 103]][0]) # 打印第一个用户的阈值
