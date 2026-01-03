"""
Src/Optimizer/PPO/agent.py
"""

import torch
import torch.nn.functional as F

from Src.paras import *
from Src.Objective.objective import objective, get_lat_and_acc
from Src.Optimizer.PPO.networks import ActorCritic
from Src.Optimizer.PPO.buffer import RolloutBuffer


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
        # self.topk_buffer = TopKRolloutMemory(capacity=5)


    def sample_action(self, state):
        logits_X, mu_Y, std_Y, value = self.policy(state)

        # ===== X: hierarchical policy with k in {0,1,2}, sum(row)<=2 =====
        logits_X = torch.clamp(logits_X, -10, 10)  # (1, n*m) expected
        logits_X_row = logits_X.squeeze(0).view(self.paras.n, self.paras.m)  # (n, m)

        action_X = np.zeros((self.paras.n, self.paras.m), dtype=np.float32)
        logprob_X = torch.tensor(0.0, dtype=torch.float32, device=logits_X.device)

        for i in range(self.paras.n):
            row_logits = logits_X_row[i]  # (m,)

            # ---- 1) sample k ∈ {0,1,2} from row-derived gate ----
            s = row_logits.mean()  # scalar
            k_logits = torch.stack([torch.tensor(0.0, device=row_logits.device),
                                    s,
                                    2.0 * s])  # (3,)
            k_probs = torch.softmax(k_logits, dim=-1)
            k_dist = torch.distributions.Categorical(probs=k_probs)
            k = int(k_dist.sample().item())

            # log p(k)
            logprob_X = logprob_X + torch.log(k_probs[k] + 1e-12)

            # ---- 2) sample indices conditioned on k ----
            probs1 = torch.softmax(row_logits, dim=-1)  # (m,)

            if k == 0:
                # choose nothing
                continue

            elif k == 1:
                dist1 = torch.distributions.Categorical(probs=probs1)
                a1 = int(dist1.sample().item())
                action_X[i, a1] = 1.0
                logprob_X = logprob_X + torch.log(probs1[a1] + 1e-12)

            else:  # k == 2
                dist1 = torch.distributions.Categorical(probs=probs1)
                a1_t = dist1.sample()

                probs2 = probs1.clone()
                probs2[a1_t] = 0.0
                probs2 = probs2 / (probs2.sum() + 1e-12)
                dist2 = torch.distributions.Categorical(probs=probs2)
                a2_t = dist2.sample()

                # fixed order (ascending) for consistency with update parsing
                j1, j2 = int(a1_t.item()), int(a2_t.item())
                if j1 > j2:
                    j1, j2 = j2, j1

                action_X[i, j1] = 1.0
                action_X[i, j2] = 1.0

                # log p(j1) + log p(j2 | j1 masked), using the same fixed order
                logprob_X = logprob_X + torch.log(probs1[j1] + 1e-12)
                probs2_fix = probs1.clone()
                probs2_fix[j1] = 0.0
                probs2_fix = probs2_fix / (probs2_fix.sum() + 1e-12)
                logprob_X = logprob_X + torch.log(probs2_fix[j2] + 1e-12)

        # ===== Y: keep your original Normal sampling =====
        dist_Y = torch.distributions.Normal(mu_Y, std_Y)
        sampled_Y = dist_Y.sample()
        action_Y = sampled_Y.detach().cpu().numpy().reshape(self.paras.n, self.paras.m)

        # ===== hard constraints =====
        action_X = _project_X(action_X)  # should keep sum<=2
        action_Y = _clip_Y(action_Y, self.paras.E)
        action_Y = np.clip(action_Y, 0, 1)

        # ===== total logprob =====
        logprob_Y = dist_Y.log_prob(sampled_Y).sum()
        logprob = (logprob_X + logprob_Y).detach()

        return action_X, action_Y, logprob, value


    def update_policy(self, epoch):
        entropy_coef = self.initial_entropy_coef * (self.entropy_decay ** epoch)

        advantages, returns = self.buffer.compute_advantages(self.hparams['gamma'], self.hparams['lam'])
        # Fix: Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.stack(self.buffer.states)  # (B, state_dim)

        actions_X = torch.tensor(np.stack(self.buffer.actions_X), dtype=torch.float32).view(len(states), -1)
        actions_Y = torch.tensor(np.stack(self.buffer.actions_Y), dtype=torch.float32).view(len(states), -1)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()

        B = len(states)
        n, m = self.paras.n, self.paras.m

        for _ in range(self.hparams['k_epochs']):
            logits_X, mu_Y, std_Y, values_new = self.policy(states)
            logits_X = torch.clamp(logits_X, -10, 10)

            logits_X_rows = logits_X.view(B, n, m)  # (B, n, m)
            actions_X_rows = actions_X.view(B, n, m)  # (B, n, m)

            new_logprob_X = torch.zeros(B, dtype=torch.float32, device=logits_X.device)
            entropy_X = torch.zeros(B, dtype=torch.float32, device=logits_X.device)

            for b in range(B):
                lp_b = torch.tensor(0.0, dtype=torch.float32, device=logits_X.device)
                ent_b = torch.tensor(0.0, dtype=torch.float32, device=logits_X.device)

                for i in range(n):
                    row_logits = logits_X_rows[b, i]  # (m,)
                    probs1 = torch.softmax(row_logits, dim=-1)

                    # ---- gate distribution p(k) ----
                    s = row_logits.mean()
                    k_logits = torch.stack([torch.tensor(0.0, device=row_logits.device),
                                            s,
                                            2.0 * s])
                    k_probs = torch.softmax(k_logits, dim=-1)

                    # infer k from action row sum (threshold to avoid float noise)
                    row = actions_X_rows[b, i]
                    row_bin = (row > 0.5).to(torch.float32)
                    k = int(torch.clamp(row_bin.sum(), 0, 2).item())  # 0/1/2

                    # log p(k) and entropy H(k)
                    lp_b = lp_b + torch.log(k_probs[k] + 1e-12)
                    k_dist = torch.distributions.Categorical(probs=k_probs)
                    ent_b = ent_b + k_dist.entropy()

                    if k == 0:
                        continue

                    elif k == 1:
                        # find the single selected index
                        j = int(torch.argmax(row_bin).item())
                        lp_b = lp_b + torch.log(probs1[j] + 1e-12)

                        dist1 = torch.distributions.Categorical(probs=probs1)
                        ent_b = ent_b + dist1.entropy()

                    else:  # k == 2
                        idx = torch.topk(row_bin, k=2).indices
                        j1 = int(torch.min(idx).item())
                        j2 = int(torch.max(idx).item())

                        lp_b = lp_b + torch.log(probs1[j1] + 1e-12)

                        probs2 = probs1.clone()
                        probs2[j1] = 0.0
                        probs2 = probs2 / (probs2.sum() + 1e-12)
                        lp_b = lp_b + torch.log(probs2[j2] + 1e-12)

                        dist1 = torch.distributions.Categorical(probs=probs1)
                        dist2 = torch.distributions.Categorical(probs=probs2)
                        ent_b = ent_b + dist1.entropy() + dist2.entropy()

                new_logprob_X[b] = lp_b
                entropy_X[b] = ent_b

            # ===== Y: same as before =====
            dist_Y = torch.distributions.Normal(mu_Y, std_Y)
            new_logprob_Y = dist_Y.log_prob(actions_Y).sum(1)

            new_logprob = new_logprob_X + new_logprob_Y

            ratio = torch.exp(new_logprob - old_logprobs)
            ratio = torch.clamp(ratio, 0, 10)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.hparams['eps_clip'], 1 + self.hparams['eps_clip']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new.squeeze(), returns)

            entropy_Y = dist_Y.entropy().sum(1)
            entropy = entropy_X + entropy_Y

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
                self.buffer.add(state.squeeze(), action_X, action_Y, logprob, value.item(), reward, 1)
                if reward > best_epoch_reward:
                    best_epoch_reward = reward
                    best_epoch_action_X = action_X.copy()
                    best_epoch_action_Y = action_Y.copy()
                # 1.4 下个step
                X, Y = action_X, action_Y
                steps += 1



            # # 2. 稳定结果
            # # 2.1 最大值 回滚机制
            # rollout_data = self.buffer.get_all_data()
            # self.topk_buffer.add(rollout_data, best_epoch_reward)
            # # 2.2 top-k 拼接机制
            # extra_buffers = self.topk_buffer.get_all()
            # for extra in extra_buffers:
            #     self.buffer.extend(extra)


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
    print("todo")
