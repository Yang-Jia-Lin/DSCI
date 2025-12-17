from Src.paras import *
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Objective.objective import objective
from Src.Utils.plot_function import plot_decisions, plot_convergence
from Src.Optimizer.GA import optimize_GA


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

# Optimize
GA_best_val, GA_best_sol, GA_history = optimize_GA(paras)
GA_X_opt, GA_Y_opt, GA_F_e_opt, GA_F_c_opt = GA_best_sol
plot_convergence(GA_history, data_name="GA_Convergence_history", save_dir=Path(RESULT_GA_PATH))
plot_decisions(GA_X_opt, GA_Y_opt, data_name="GA最优结果示意图", save_dir=Path(RESULT_GA_PATH))
