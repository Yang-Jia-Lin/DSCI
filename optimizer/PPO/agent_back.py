import torch
import torch.nn.functional as F
import numpy as np

from objective.compute_P import compute_layer_exit_probs
from optimizer.PPO.networks import ActorCritic
from optimizer.PPO.buffer import RolloutBuffer
from optimizer.PPO.environment import project_X, clip_Y, compute_reward, init_feasible_XY
from objective.objective import objective
from utils.compute_paras import compute_iota_kappa, allocate_resources


class PPOAgent:
    def __init__(self, paras, hyperparams):
        self.paras = paras
        self.hparams = hyperparams

        n, m = paras.n, paras.m
        self.state_dim = n * m * 2
        self.action_dim_X = n * m
        self.action_dim_Y = n * m

        self.policy = ActorCritic(self.state_dim, self.action_dim_X, self.action_dim_Y)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperparams['lr'])
        self.buffer = RolloutBuffer()

    def flatten_state(self, X, Y):
        return torch.tensor(np.concatenate([X.flatten(), Y.flatten()]), dtype=torch.float32).unsqueeze(0)

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
        action_X = project_X(action_X)
        action_Y = clip_Y(action_Y, self.paras.E)
        action_Y = np.clip(action_Y, 0, 1)  # 二次clip保证

        # logprob & value
        dist_X = torch.distributions.Bernoulli(torch.clamp(torch.sigmoid(logits_X), 1e-6, 1-1e-6))
        logprob_X = dist_X.log_prob(torch.tensor(action_X.flatten(), dtype=torch.float32)).sum()

        logprob_Y = dist_Y.log_prob(sampled_Y).sum()

        logprob = (logprob_X + logprob_Y).detach()

        return action_X, action_Y, logprob, value

    def update_policy(self):
        if len(self.buffer.rewards) == 0:
            print("[Warning] update_policy called with empty buffer, skip.")
            return

        advantages, returns = self.buffer.compute_advantages(self.hparams['gamma'], self.hparams['lam'])
        states = torch.stack(self.buffer.states)
        actions_X = torch.tensor(np.stack(self.buffer.actions_X), dtype=torch.float32).view(len(states), -1)
        actions_Y = torch.tensor(np.stack(self.buffer.actions_Y), dtype=torch.float32).view(len(states), -1)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()

        for _ in range(self.hparams['k_epochs']):
            logits_X, mu_Y, std_Y, values_new = self.policy(states)
            logits_X = torch.clamp(logits_X, -10, 10)
            p_X_probs = torch.clamp(torch.sigmoid(logits_X), 1e-6, 1-1e-6)

            dist_X = torch.distributions.Bernoulli(p_X_probs)
            dist_Y = torch.distributions.Normal(mu_Y, std_Y)

            new_logprob_X = dist_X.log_prob(actions_X).sum(1)
            new_logprob_Y = dist_Y.log_prob(actions_Y).sum(1)
            new_logprob = new_logprob_X + new_logprob_Y

            ratio = torch.exp(new_logprob - old_logprobs)
            ratio = torch.clamp(ratio, 0, 10)  # 防止梯度爆炸

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.hparams['eps_clip'], 1 + self.hparams['eps_clip']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new.squeeze(), returns)
            total_loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.buffer.clear()

    def train(self):
        best_val = -np.inf
        best_sol = None
        history = []

        patience = 10
        tolerance = 1e-3

        # 初始化资源
        F_e = np.ones((self.paras.n, 1)) * self.paras.f_e_max / self.paras.n
        F_c = np.ones((self.paras.n, 1)) * self.paras.f_c_max / self.paras.n

        for epoch in range(self.hparams['max_epochs']):
            self.buffer.clear()

            X, Y = init_feasible_XY(self.paras)

            # 内层 PPO rollout
            steps = 0
            while steps < self.hparams['target_steps']:
                state = self.flatten_state(X, Y)
                action_X, action_Y, logprob, value = self.sample_action(state)

                reward = compute_reward(action_X, action_Y, F_e, F_c, self.paras)
                if np.isnan(reward) or np.isinf(reward):
                    reward = -1e6

                self.buffer.add(state.squeeze(), action_X, action_Y, logprob, value.item(), reward, 0)
                X, Y = action_X, action_Y
                steps += 1

            # ===============================
            # 🌟 外层：凸优化 f_e, f_c
            # ===============================
            X_cur, Y_cur = X, Y  # 用内层 PPO 最优 X, Y
            P = compute_layer_exit_probs(Y_cur, self.paras)
            iota, kappa = compute_iota_kappa(X_cur, self.paras.C, P)
            F_e, F_c = allocate_resources(iota, kappa, self.paras.f_e_max, self.paras.f_c_max)
            print(f"F_e: {sum(F_e)}, F_c: {sum(F_c)}")
            if np.any(np.isnan(F_e)) or np.any(np.isinf(F_e)) or np.any(np.isnan(F_c)) or np.any(np.isinf(F_c)):
                print("[Warning] F_e or F_c contains invalid values. Reset to uniform.")
                F_e = np.ones((self.paras.n, 1)) * self.paras.f_e_max / self.paras.n
                F_c = np.ones((self.paras.n, 1)) * self.paras.f_c_max / self.paras.n

            rewards_arr = np.array(self.buffer.rewards, dtype=np.float32)
            rewards_arr = rewards_arr[np.isfinite(rewards_arr)]
            if len(rewards_arr) == 0:
                print(f"[Warning] Empty or invalid rewards at epoch {epoch}, skip max computation.")
                current_best = -np.inf
            else:
                current_best = np.max(rewards_arr)

            history.append(current_best)
            if current_best > best_val:
                best_val = current_best
                best_sol = (X, Y, F_e, F_c)

            print(f"Epoch {epoch}: best_val={best_val}")

            # 收敛检测
            if len(history) > patience:
                recent_window = history[-patience:]
                if np.std(recent_window) < tolerance:
                    print(f"[Early Stop] Converged at epoch {epoch} with std: {np.std(recent_window):.5f}")
                    break

            self.update_policy()
        return best_val, best_sol, history
