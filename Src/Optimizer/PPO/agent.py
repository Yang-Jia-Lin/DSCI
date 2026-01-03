"""
Src/Optimizer/PPO/agent.py
"""

import torch
import torch.nn.functional as F

from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.compute_accuracy import compute_expected_accuracy
from Src.Objective.compute_latency import compute_total_latency
from Src.Objective.objective import objective, get_lat_and_acc
from Src.Optimizer.PPO.networks import ActorCritic
from Src.Optimizer.PPO.buffer import RolloutBuffer, TopKRolloutMemory


# 约束 X
def _project_X(X):
    """ 保证 X 每行最多2个1，其余置0"""
    for i in range(X.shape[0]):
        if np.sum(X[i]) > 2:
            # 直接随机保留2个
            ones_idx = np.where(X[i]==1)[0]
            keep_idx = np.random.choice(ones_idx, size=2, replace=False)
            X[i] = 0
            X[i][keep_idx] = 1
    return X


# 约束 Y
def _clip_Y(Y, E):
    """ 对 Y clip到[0,1]，并固定不可早退层(j not in E)为1 """
    Y = np.clip(Y, 0, 1)
    all_indices = np.arange(Y.shape[1])
    fixed_indices = np.setdiff1d(all_indices, E)
    Y[:, fixed_indices] = 1.0
    return Y


# 计算奖励
def _compute_reward(X,Y,F_e,F_c,paras):
    reward = objective(X,Y,F_e,F_c,paras)
    if np.isnan(reward) or np.isinf(reward):
         print(f"【Warning】: reward is nan or inf: {reward}\naction_X: {X}, action_Y: {Y}, f_e: {F_e}, f_c: {F_c}")
         reward = 0
    return reward


# 随机初始化
def _init_feasible_XY(paras):
    n, m = paras.n, paras.m
    X = np.random.randint(0, 2, (n, m))
    X = _project_X(X)
    Y = np.random.uniform(0, 1, (n, m))
    Y = _clip_Y(Y, paras.E)
    return X, Y

# 展平 X+Y
def _flatten_state(X, Y):
    return torch.tensor(np.concatenate([X.flatten(), Y.flatten()]), dtype=torch.float32).unsqueeze(0)


class PPOAgent:
    def __init__(self, paras, hyperparams):
        self.paras = paras
        self.hparams = hyperparams
        self.initial_entropy_coef = hyperparams.get('entropy_coef', 0.01)
        self.entropy_decay = hyperparams.get('entropy_decay', 0.99)

        n, m = paras.n, paras.m
        self.state_dim = n * m * 2
        self.action_dim_X = n * m
        self.action_dim_Y = n * m

        self.policy = ActorCritic(self.state_dim, self.action_dim_X, self.action_dim_Y)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperparams['lr'])
        self.buffer = RolloutBuffer()

        self.best_policy_state_dict = None  # 保存策略参数
        self.topk_buffer = TopKRolloutMemory(capacity=5)

    def sample_action(self, state):
        logits_X, mu_Y, std_Y, value = self.policy(state)

        # 对X: Bernoulli采样并按概率保留2个1
        logits_X = torch.clamp(logits_X, -10, 10)
        p_X = torch.sigmoid(logits_X).detach().cpu().numpy().reshape(self.paras.n, self.paras.m)
        action_X = np.zeros_like(p_X)
        for i in range(action_X.shape[0]):
            top2_idx = np.argsort(p_X[i])[-2:]
            action_X[i][top2_idx] = 1

        # 对Y: Normal采样
        dist_Y = torch.distributions.Normal(mu_Y, std_Y)
        sampled_Y = dist_Y.sample()
        action_Y = sampled_Y.detach().cpu().numpy().reshape(self.paras.n, self.paras.m)

        # 执行环境硬约束
        action_X = _project_X(action_X)
        action_Y = _clip_Y(action_Y, self.paras.E)
        action_Y = np.clip(action_Y, 0, 1)  # 二次clip保证

        # logprob & value
        dist_X = torch.distributions.Bernoulli(torch.clamp(torch.sigmoid(logits_X), 1e-6, 1 - 1e-6))
        logprob_X = dist_X.log_prob(torch.tensor(action_X.flatten(), dtype=torch.float32)).sum()
        logprob_Y = dist_Y.log_prob(sampled_Y).sum()
        logprob = (logprob_X + logprob_Y).detach()

        return action_X, action_Y, logprob, value


    def update_policy(self, epoch):
        entropy_coef = self.initial_entropy_coef * (self.entropy_decay ** epoch)

        advantages, returns = self.buffer.compute_advantages(self.hparams['gamma'], self.hparams['lam'])
        states = torch.stack(self.buffer.states)
        actions_X = torch.tensor(np.stack(self.buffer.actions_X), dtype=torch.float32).view(len(states), -1)
        actions_Y = torch.tensor(np.stack(self.buffer.actions_Y), dtype=torch.float32).view(len(states), -1)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()

        for _ in range(self.hparams['k_epochs']):
            logits_X, mu_Y, std_Y, values_new = self.policy(states)
            logits_X = torch.clamp(logits_X, -10, 10)
            p_X_probs = torch.clamp(torch.sigmoid(logits_X), 1e-6, 1 - 1e-6)

            dist_X = torch.distributions.Bernoulli(p_X_probs)
            dist_Y = torch.distributions.Normal(mu_Y, std_Y)

            new_logprob_X = dist_X.log_prob(actions_X).sum(1)
            new_logprob_Y = dist_Y.log_prob(actions_Y).sum(1)
            new_logprob = new_logprob_X + new_logprob_Y

            ratio = torch.exp(new_logprob - old_logprobs)
            ratio = torch.clamp(ratio, 0, 10)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.hparams['eps_clip'], 1 + self.hparams['eps_clip']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new.squeeze(), returns)

            # 动态熵
            entropy = dist_X.entropy().sum(1) + dist_Y.entropy().sum(1)
            total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


    def train(self):
        best_val = -np.inf
        best_sol = None
        history = []

        patience = 10
        tolerance = 1e-3

        # 初始化资源
        F_e = np.ones((self.paras.n, 1)) * self.paras.f_e_max / self.paras.n # GHz
        F_c = np.ones((self.paras.n, 1)) * self.paras.f_c_max / self.paras.n # GHz

        for epoch in range(self.hparams['max_epochs']):

            # ========== 内层：强化学习 ==========
            # 0. 初始化 X, Y, result, buffer
            X, Y = _init_feasible_XY(self.paras)
            best_epoch_reward = -np.inf
            best_epoch_action_X, best_epoch_action_Y = X, Y
            self.buffer.clear()  # 每个 epoch 清空上一轮的所有经验，重新总结


            # 1. 多次动作 采样到buffer中（Rollout: target_steps）
            steps = 0
            while steps < self.hparams['target_steps']:
                state = _flatten_state(X, Y)
                # 1.1 执行动作
                action_X, action_Y, logprob, value = self.sample_action(state)
                # 1.2 计算奖励
                reward = _compute_reward(action_X, action_Y, F_e, F_c, self.paras)
                # 1.3 记录到buffer
                self.buffer.add(state.squeeze(), action_X, action_Y, logprob, value.item(), reward, 0)
                if reward > best_epoch_reward:
                    best_epoch_reward = reward
                    best_epoch_action_X = action_X.copy()
                    best_epoch_action_Y = action_Y.copy()
                # 1.4 下个step
                X, Y = action_X, action_Y
                steps += 1



            # 2. 稳定结果
            # 2.1 最大值 回滚机制
            rollout_data = self.buffer.get_all_data()
            self.topk_buffer.add(rollout_data, best_epoch_reward)
            # 2.2 top-k 拼接机制
            extra_buffers = self.topk_buffer.get_all()
            for extra in extra_buffers:
                self.buffer.extend(extra)


            # 3. 根据多步采样结果更新策略
            self.update_policy(epoch)


            # ========== 外层：凸优化 ==========
            # P = compute_layer_exit_probs(best_epoch_action_Y, self.paras)
            # iota, kappa = compute_iota_kappa(best_epoch_action_X, self.paras.C, P)
            # F_e, F_c = allocate_resources(iota, kappa, self.paras.f_e_max, self.paras.f_c_max)
            # if np.any(np.isnan(F_e)) or np.any(np.isinf(F_e)) or np.any(np.isnan(F_c)) or np.any(np.isinf(F_c)):
            #     print("【Warning】 F_e or F_c is invalid values.")
            #     F_e = np.ones((self.paras.n, 1)) * self.paras.f_e_max / self.paras.n
            #     F_c = np.ones((self.paras.n, 1)) * self.paras.f_c_max / self.paras.n


            # =========== 统计结果 ==========
            # 整体表现
            best_epoch_reward = _compute_reward(best_epoch_action_X, best_epoch_action_Y, F_e, F_c, self.paras)
            history.append(best_epoch_reward)
            if best_epoch_reward > best_val:
                best_val = best_epoch_reward
                best_sol = (best_epoch_action_X, best_epoch_action_Y, F_e, F_c)
                self.best_policy_state_dict = {k: v.clone() for k, v in self.policy.state_dict().items()}

            # 具体时延和精度
            latency, acc = get_lat_and_acc(best_epoch_action_X, best_epoch_action_Y, F_e, F_c, self.paras)
            print(f"Epoch {epoch}: current_val={best_epoch_reward}, latency={latency}, acc={acc}")

            # 回滚
            if epoch > 5 and best_epoch_reward < np.mean(history[-5:]) - 1:
                print(f"[Rollback] Performance degraded at epoch {epoch}, rolling back best policy.")
                self.policy.load_state_dict(self.best_policy_state_dict)

            # ========== 收敛检测 ==========
            if len(history) > patience:
                recent_window = history[-patience:]
                if np.std(recent_window) < tolerance:
                    print(f"[Early Stop] Converged at epoch {epoch} with std: {np.std(recent_window):.5f}")
                    break

        return best_val, best_sol, history



if __name__ == "__main__":
    from Src.paras import *
    from Src.Utils.parsing_data import parsing_rate_and_acc
    from Src.Utils.plot_function import plot_convergence
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
        'max_epochs': 10,  # 💥 只跑10个epoch
        'target_steps': 512,  # 少采样一点
        'k_epochs': 2
    }

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
