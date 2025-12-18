from Src.paras import *
from Src.Optimizer.BF.alg_BF import optimize_BF
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Utils.log_function import save_experiment_results


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


# Optimize
bf_params = {'max_iter': 5, 'restarts': 2, 'threshold_step': 0.1}
BF_best_val, BF_best_sol, BF_history = optimize_BF(
    paras,
    max_iter=bf_params['max_iter'],
    restarts=bf_params['restarts'],
    threshold_step=bf_params['threshold_step']
)


# Log
save_experiment_results(
    save_dir=Path(RESULT_BF_PATH),
    algo_name="BF",
    paras=paras,
    best_val=BF_best_val,
    best_sol=BF_best_sol,
    history=BF_history,
    hyper_params=bf_params
)
