"""
Src/Optimizer/PPO/run_PPO.py
"""

from Src.paras import *
from Src.Utils.parsing_data import parsing_rate_and_acc
from Src.Optimizer.PPO.agent import PPOAgent
from Src.Utils.log_function import save_experiment_results


def run_dsci_experiment(custom_paras_dict=None, custom_ppo_hyperparams=None, save_log=True):
    """
    封装 DSCI 运行逻辑，支持动态参数注入
    """
    # 1. 初始化基础参数
    paras = Paras(
        n=NUM_USERS, m=NUM_LAYERS, E=EARLY_EXIT_LAYERS, D=DATA_SIZE_LAYERS, C=COMPUTE_SIZE_LAYERS,
        G=BASE_STATION_POWER, delta=NOISE_POWER, alpha=1, beta=5,
        f_e_max=EDGE_MAX_FREQ, f_c_max=CLOUD_MAX_FREQ, H_u=CHANNEL_GAINS_USERS,
        F_u=USER_FREQs, b_e=BANDWIDTH_EDGE, b_c=BANDWIDTH_CLOUD
    )

    # 2. 动态实验参数 (例如修改 F_u 或 b_e)
    if custom_paras_dict:
        for key, value in custom_paras_dict.items():
            setattr(paras, key, value)
    # 参数更新后必须重新计算通信速率
    paras.rates, paras.accs = parsing_rate_and_acc(paras)

    # 3. 设置 PPO 超参数
    ppo_params = {
        'gamma': 0.95,
        'lam': 0.95,
        'lr': 1e-4,
        'eps_clip': 0.15,
        'max_epochs': 200,
        'target_steps': 1024,
        'k_epochs': 10,
        'entropy_coef': 0.01,
        'entropy_decay': 0.995,
        'grad_clip': 0.5,
        'obj_scale': 1000.0
    }
    if custom_ppo_hyperparams:
        ppo_params.update(custom_ppo_hyperparams)

    # 4. 算法优化
    agent = PPOAgent(paras, ppo_params)
    best_val, best_sol, history = agent.train()

    # 5. 日志保存
    if save_log:
        save_experiment_results(
            save_dir=Path(RESULT_PPO_PATH),
            algo_name="PPO_Exp",
            paras=paras,
            best_val=best_val,
            best_sol=best_sol,
            history=history,
            hyper_params=ppo_params,
            extra_logs=agent.logs
        )

    return best_val, best_sol, history


if __name__ == '__main__':
    run_dsci_experiment(save_log=True)
