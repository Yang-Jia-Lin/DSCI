"""
Src/Optimizer/PPO/run_PPO.py
"""

from Src.paras import *
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Optimizer.PPO.agent import PPOAgent
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
    beta=5
)
paras.rates, paras.accs = parsing_rate_and_acc(paras)


# Optimize
PPO_hyperparams = {
    'gamma': 0.95,
    'lam': 0.95,
    'lr': 1e-4,
    'eps_clip': 0.15,
    'max_epochs': 100,
    'target_steps': 1024,
    'k_epochs': 10,
    'entropy_coef': 0.01,
    'entropy_decay': 0.995,
    'grad_clip': 0.5,
    'obj_scale': 1000.0
}
PPO_best_val, PPO_best_sol, PPO_history = PPOAgent(
    paras,
    PPO_hyperparams
).train()


# Log
save_experiment_results(
    save_dir=Path(RESULT_PPO_PATH),
    algo_name="PPO",
    paras=paras,
    best_val=PPO_best_val,
    best_sol=PPO_best_sol,
    history=PPO_history,
    hyper_params=PPO_hyperparams
)
