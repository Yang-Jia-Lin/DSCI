# optimize_PPO() 的主入口

from optimizer.PPO.agent import PPOAgent

hyperparams = {
    'gamma': 0.99,
    'lam': 0.95,
    'lr': 1e-4,
    'eps_clip': 0.15,
    'max_epochs': 100,
    'target_steps': 2048,
    'k_epochs': 10,
    'entropy_coef': 0.5,
    'entropy_decay': 0.999
}


def optimize_PPO(paras):
    agent = PPOAgent(paras, hyperparams)
    best_val, best_sol, history = agent.train()
    return best_val, best_sol, history

