# Src/Optimizer/GA/run_GA.py

from Src.paras import *
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Optimizer.GA.alg_GA import optimize_GA
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
GA_hyperparams = {
    'population_size': 50,
    'generations': 150,
    'mutation_rate': 0.1
}
GA_best_val, GA_best_sol, GA_history = optimize_GA(
    paras,
    population_size = GA_hyperparams['population_size'],
    generations = GA_hyperparams['generations'],
    mutation_rate = GA_hyperparams['mutation_rate']
)


# Log
save_experiment_results(
    save_dir=Path(RESULT_GA_PATH),
    algo_name="GA",
    paras=paras,
    best_val=GA_best_val,
    best_sol=GA_best_sol,
    history=GA_history,
    hyper_params=GA_hyperparams
)