"""
quick_test_PPO.py

这是一个极简 smoke test，用于快速检查 PPO 模块是否能正常运行。
只跑 10 个 epoch，并打印主要日志与收敛趋势。
"""

from paras import *
from utils.parsing_data import parsing_rate_and_acc
from utils.plot_results import plot_convergence
from pathlib import Path


# ============ 参数准备 ================
paras = Paras(
    n=NUM_USERS,
    m=NUM_LAYERS,
    E=EARLY_EXIT_LAYERS,
    D=DATA_SIZE_LAYERS,
    C=COMPUTE_SIZE_LAYERS,
    F_u=USER_FREQs,
    f_e_max=EDGE_MAX_FREQ,
    f_c_max=CLOUD_MAX_FREQ,
    H_u=CHANNEL_GAINS_USERS,
    b_e=BANDWIDTH_EDGE,
    b_c=BANDWIDTH_CLOUD,
    G=BASE_STATION_POWER,
    delta=NOISE_POWER,
    alpha=100000,
    beta=1
)
paras.rates, paras.accs = parsing_rate_and_acc(paras)

# ============ 一键 PPO 测试 =============
print("🚀 Starting quick PPO test...")

# 只运行10个epoch
hyperparams = {
    'gamma': 0.99,
    'lam': 0.95,
    'lr': 3e-4,
    'eps_clip': 0.2,
    'max_epochs': 10,     # 💥 只跑10个epoch
    'target_steps': 512,  # 少采样一点
    'k_epochs': 2
}

from optimizer.PPO.agent import PPOAgent

agent = PPOAgent(paras, hyperparams)
best_val, best_sol, history = agent.train()

print("✅ PPO quick test completed.")
print(f"Best objective: {best_val}")
print(f"History (last 5): {history[-5:]}")

# ============ 绘图保存 =============
output_dir = Path("./quick_test_results")
output_dir.mkdir(exist_ok=True)

plot_convergence(history, data_name="Quick_PPO_Convergence", save_dir=output_dir)
print(f"📈 Plot saved to {output_dir / 'Quick_PPO_Convergence.png'}")
