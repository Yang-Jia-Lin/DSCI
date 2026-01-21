"""
Src/Optimizer/PPO/agent.py

适配改动：
1) Episode 包含 n 个 steps（每步一个用户）
2) X: categorical index over all valid (k1,k2) pairs（来自 network.x_pairs）
3) Y: Beta 分布，仅对早退层集合 |E| 输出/采样（无需硬裁剪）
4) 数值稳定：adv norm、grad clip、严格 on-policy，移除 TopK/off-policy 等机制
"""

import numpy as np
import torch
import torch.nn.functional as F

from Src.Objective.objective import objective, get_lat_and_acc
from Src.Optimizer.PPO.networks import ActorCritic
from Src.Optimizer.PPO.buffer import RolloutBuffer


# ---------- 状态构造（紧凑 Markov） ----------
def _build_state(i: int, n: int, prev_obj: float, obj_scale: float = 1000.0) -> torch.Tensor:
    """
    state = [i_norm, remaining_norm, tanh(prev_obj/scale)]
    - i_norm: 当前用户索引归一化
    - remaining_norm: 剩余用户比例（作为“remaining_resource/remaining_steps”的简化版本）
    - prev_obj: 当前已决策到 i-1 的全局目标值（用 tanh 压缩避免数值爆）
    """
    i_norm = float(i) / float(max(n, 1))
    remaining_norm = float(n - i) / float(max(n, 1))
    prev_obj_squashed = np.tanh(prev_obj / obj_scale)
    s = torch.tensor([i_norm, remaining_norm, prev_obj_squashed], dtype=torch.float32).unsqueeze(0)
    return s


# ---------- 初始化一个可行解（给未决策用户用作 baseline） ----------
def _init_feasible_XY(paras):
    """
    生成一个“默认可行”的 X, Y，用作 episode 初始基线和未决策用户的占位。
    - X: 每行两个切分点 (k1,k2)，这里用 (m//3, 2m//3)
    - Y: 全 1，早退层也先设为 1（表示阈值高，倾向不早退；具体语义由你的 compute_* 决定）
    """
    n, m = paras.n, paras.m
    X = np.zeros((n, m), dtype=np.float32)
    k1 = max(0, m // 3)
    k2 = min(m - 1, (2 * m) // 3)
    if k1 == k2:
        k2 = min(m - 1, k1 + 1)

    for i in range(n):
        X[i, k1] = 1.0
        X[i, k2] = 1.0

    Y = np.ones((n, m), dtype=np.float32)
    # 早退层也先设 1（不强制），RL 会学到更优的阈值
    for ee in paras.E:
        if 0 <= ee < m:
            Y[:, ee] = 1.0
    return X, Y


class PPOAgent:
    def __init__(self, paras, hyperparams):
        self.paras = paras
        self.hparams = hyperparams

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 熵系数衰减
        self.initial_entropy_coef = hyperparams.get("entropy_coef", 0.01)
        self.entropy_decay = hyperparams.get("entropy_decay", 0.99)

        # ---------- 新维度定义 ----------
        # 状态：紧凑 3 维（见 _build_state）
        self.state_dim = 3
        # 动作 Y：只对早退层输出（|E|）
        self.action_dim_Y = len(self.paras.E)

        # ---------- 网络：X categorical over all (k1,k2), Y beta(alpha,beta) ----------
        self.policy = ActorCritic(
            state_dim=self.state_dim,
            num_layers=self.paras.m,
            action_dim_Y=self.action_dim_Y,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperparams["lr"])
        self.buffer = RolloutBuffer()

        # 保存历史最优策略（用于最终 best checkpoint），不做频繁 rollback
        self.best_policy_state_dict = None

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor):
        """
        Args:
            state: [1, state_dim] on device
        Returns:
            x_idx: LongTensor scalar（categorical index）
            y: Tensor[|E|]（Beta sample in [0,1]）
            logprob: Tensor scalar（logp_X + logp_Y）
            value: Tensor scalar
        """
        logits_X, alpha_Y, beta_Y, value = self.policy(state)

        # X: categorical
        dist_X = torch.distributions.Categorical(logits=logits_X)
        x_idx = dist_X.sample()  # shape [1]
        logp_X = dist_X.log_prob(x_idx)  # shape [1]
        ent_X = dist_X.entropy()         # shape [1]

        # Y: Beta（可能 |E|=0）
        if self.action_dim_Y > 0:
            dist_Y = torch.distributions.Beta(alpha_Y, beta_Y)
            y = dist_Y.sample()  # [1, |E|]
            logp_Y = dist_Y.log_prob(y).sum(-1)  # [1]
            ent_Y = dist_Y.entropy().sum(-1)     # [1]
        else:
            y = state.new_zeros((1, 0))
            logp_Y = state.new_zeros((1,))
            ent_Y = state.new_zeros((1,))

        logprob = (logp_X + logp_Y).detach().squeeze(0)  # scalar
        value = value.detach().view(-1)[0]              # scalar
        entropy = (ent_X + ent_Y).detach().squeeze(0)   # scalar（可选返回）

        return x_idx.view(-1)[0], y.squeeze(0), logprob, value, entropy

    def _apply_action_to_XY(self, X: np.ndarray, Y: np.ndarray, user_i: int, x_idx: int, y_vec: np.ndarray):
        """
        将 (x_idx, y_vec) 写入第 user_i 行的 X,Y（其余用户保持原样）
        - x_idx -> (k1,k2) 通过 policy.x_pairs 映射
        - y_vec 写入早退层集合 paras.E 对应的位置
        """
        n, m = self.paras.n, self.paras.m
        assert 0 <= user_i < n

        # ---- 写 X 行：清空后置 2 个切分点 ----
        X[user_i, :] = 0.0
        pair = self.policy.x_pairs[x_idx].detach().cpu().numpy().tolist()  # [k1,k2]
        k1, k2 = int(pair[0]), int(pair[1])
        X[user_i, k1] = 1.0
        X[user_i, k2] = 1.0

        # ---- 写 Y 行：默认全 1，只写早退层阈值 ----
        Y[user_i, :] = 1.0
        if len(self.paras.E) > 0:
            for j, layer_idx in enumerate(self.paras.E):
                if 0 <= layer_idx < m:
                    Y[user_i, layer_idx] = float(y_vec[j])

        return X, Y

    def update_policy(self, epoch: int):
        entropy_coef = self.initial_entropy_coef * (self.entropy_decay ** epoch)

        advantages, returns = self.buffer.compute_advantages(self.hparams["gamma"], self.hparams["lam"])
        if advantages.numel() == 0:
            return

        # advantage 标准化（降方差）
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)  # 关键：避免 T=1 时 NaN
        if torch.isfinite(adv_std) and adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean

        # 在训练前检查 advantages / returns 是否有限：
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            print("[Warning] Non-finite advantages/returns, skip update.")
            return

        data = self.buffer.as_tensors(device=self.device)
        states = data["states"]                    # [T, state_dim]
        actions_X = data["actions_X"]              # [T]
        actions_Y = data["actions_Y"]              # [T, |E|]
        old_logprobs = data["logprobs"].detach()   # [T]
        returns = returns.to(self.device)          # [T]
        advantages = advantages.to(self.device)    # [T]

        for _ in range(self.hparams["k_epochs"]):
            logits_X, alpha_Y, beta_Y, values_new = self.policy(states)  # logits_X [T,num_pairs]
            values_new = values_new.view(-1)  # [T]

            # X 分布
            dist_X = torch.distributions.Categorical(logits=logits_X)
            logp_X = dist_X.log_prob(actions_X)       # [T]
            ent_X = dist_X.entropy()                  # [T]

            # Y 分布（Beta）
            if self.action_dim_Y > 0:
                dist_Y = torch.distributions.Beta(alpha_Y, beta_Y)
                logp_Y = dist_Y.log_prob(actions_Y).sum(-1)   # [T]
                ent_Y = dist_Y.entropy().sum(-1)              # [T]
            else:
                logp_Y = torch.zeros_like(logp_X)
                ent_Y = torch.zeros_like(ent_X)

            new_logprob = logp_X + logp_Y  # [T]
            entropy = ent_X + ent_Y        # [T]

            # PPO ratio
            ratio = torch.exp(new_logprob - old_logprobs)  # [T]
            # 轻微 clamp 防止极端爆炸（可选）
            ratio = torch.clamp(ratio, 0.0, 10.0)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.hparams["eps_clip"], 1 + self.hparams["eps_clip"]) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_new, returns)

            total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy.mean()

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

    def train(self):
        best_val = -np.inf
        best_sol = None
        history = []

        patience = 10
        tolerance = 1e-3

        # 初始化资源
        F_e = np.ones((self.paras.n, 1), dtype=np.float32) * (self.paras.f_e_max / self.paras.n)
        F_c = np.ones((self.paras.n, 1), dtype=np.float32) * (self.paras.f_c_max / self.paras.n)

        target_steps = int(self.hparams["target_steps"])

        for epoch in range(self.hparams["max_epochs"]):
            self.buffer.clear()

            best_epoch_obj = -np.inf
            best_epoch_X = None
            best_epoch_Y = None

            steps = 0
            while steps < target_steps:
                # ---- 新 episode：以 baseline X,Y 开始 ----
                X, Y = _init_feasible_XY(self.paras)
                prev_obj = objective(X, Y, F_e, F_c, self.paras)

                # episode 长度 = n（每步决策一个用户）
                for i in range(self.paras.n):
                    if steps >= target_steps:
                        break

                    state = _build_state(i=i, n=self.paras.n, prev_obj=prev_obj).to(self.device)

                    x_idx, y_vec_t, logprob, value, _entropy = self.sample_action(state)

                    # 应用动作到第 i 个用户
                    X_new = X.copy()
                    Y_new = Y.copy()
                    y_vec_np = y_vec_t.detach().cpu().numpy().astype(np.float32)
                    X_new, Y_new = self._apply_action_to_XY(
                        X_new, Y_new, user_i=i, x_idx=int(x_idx.item()) if isinstance(x_idx, torch.Tensor) else int(x_idx),
                        y_vec=y_vec_np
                    )

                    # 增量奖励：r_t = U(s_{t+1}) - U(s_t)
                    new_obj = objective(X_new, Y_new, F_e, F_c, self.paras)
                    reward = float(new_obj - prev_obj)

                    done = 1.0 if (i == self.paras.n - 1) else 0.0

                    # 存 buffer（buffer 存的是：state(1,3)->squeeze, x_idx, y_vec, logprob, value, reward, done）
                    self.buffer.add(
                        state.squeeze(0).detach().cpu(),
                        int(x_idx.item()) if isinstance(x_idx, torch.Tensor) else int(x_idx),
                        torch.tensor(y_vec_np, dtype=torch.float32),
                        logprob.detach().cpu(),
                        float(value.item()) if isinstance(value, torch.Tensor) else float(value),
                        reward,
                        done
                    )

                    # 状态推进
                    X, Y = X_new, Y_new
                    prev_obj = new_obj
                    steps += 1

                # episode 结束：记录本 epoch 中最好的 full objective（不是增量）
                final_obj = prev_obj
                if final_obj > best_epoch_obj:
                    best_epoch_obj = final_obj
                    best_epoch_X = X.copy()
                    best_epoch_Y = Y.copy()

            # 用 rollout 更新策略
            self.update_policy(epoch)

            # 统计与 best checkpoint
            history.append(best_epoch_obj)
            if best_epoch_obj > best_val:
                best_val = best_epoch_obj
                best_sol = (best_epoch_X, best_epoch_Y, F_e, F_c)
                self.best_policy_state_dict = {k: v.clone() for k, v in self.policy.state_dict().items()}

            latency, acc = get_lat_and_acc(best_epoch_X, best_epoch_Y, F_e, F_c, self.paras)
            print(f"Epoch {epoch}: best_obj={best_epoch_obj}, latency={latency}, acc={acc}")

            # 收敛检测（窗口内波动很小就停）
            if len(history) > patience:
                recent_window = history[-patience:]
                if np.std(recent_window) < tolerance:
                    print(f"[Early Stop] Converged at epoch {epoch} with std: {np.std(recent_window):.5f}")
                    break

        return best_val, best_sol, history


if __name__ == "__main__":
    # ========== Minimal unit test for PPOAgent ==========
    # 目的：不依赖外部 CSV / parsing_rate_and_acc，仅验证 network->agent->buffer->update->train 能跑通

    import numpy as np
    from dataclasses import dataclass

    # ---- 1) 构造一个最小 Paras（只保留 agent/objective 可能会用到的字段） ----
    @dataclass
    class _TestParas:
        n: int
        m: int
        E: list
        D: list
        C: list
        F_u: np.ndarray
        f_e_max: float
        f_c_max: float
        H_u: np.ndarray
        b_e: float
        b_c: float
        G: float
        delta: float
        alpha: float
        beta: float

    # 小规模配置
    n = 4
    m = 16
    E = [5, 10]  # 两个早退层
    D = [1] * m
    C = [1] * m

    paras = _TestParas(
        n=n,
        m=m,
        E=E,
        D=D,
        C=C,
        F_u=np.ones(n, dtype=np.float32) * 2.0,
        f_e_max=10.0,
        f_c_max=20.0,
        H_u=np.ones(n, dtype=np.float32) * 2.0,
        b_e=10.0,
        b_c=20.0,
        G=1.0,
        delta=1e-9,
        alpha=1.0,
        beta=1.0,
    )

    # ---- 2) Monkey patch: 用假的 objective/get_lat_and_acc，避免依赖你真实 Objective 模块 ----
    # 这个假的目标函数只保证：
    # - 随 X/Y 改变而变化（有梯度学习意义）
    # - 数值稳定、运行快
    #
    # U = +0.1 * sum(Y at early exits) - 0.01 * sum(split positions)
    # （仅用于跑通流程，不代表真实性能）
    def _fake_objective(X, Y, F_e, F_c, paras_):
        # X: [n,m] one-hot at two split points each row
        # Y: [n,m], early exit thresholds at E positions; others 1
        x_cost = float(np.sum(np.argmax(X, axis=1)))  # 用一个简单的“位置成本”
        y_gain = float(np.sum(Y[:, paras_.E])) if len(paras_.E) > 0 else 0.0
        return 0.1 * y_gain - 0.01 * x_cost

    def _fake_get_lat_and_acc(X, Y, F_e, F_c, paras_):
        # 给出两个可打印指标
        latency = float(np.sum(np.argmax(X, axis=1)))  # 越靠后越大
        acc = float(np.mean(Y[:, paras_.E])) if len(paras_.E) > 0 else 1.0
        return latency, acc

    # 把当前模块里 import 进来的 objective/get_lat_and_acc 覆盖掉
    globals()["objective"] = _fake_objective
    globals()["get_lat_and_acc"] = _fake_get_lat_and_acc

    # ---- 3) 超参：极小规模，快速验证 ----
    hparams = {
        "gamma": 0.95,
        "lam": 0.95,
        "lr": 1e-4,
        "eps_clip": 0.2,
        "max_epochs": 2,
        "target_steps": 32,   # 小一点就行
        "k_epochs": 2,
        "entropy_coef": 0.01,
        "entropy_decay": 1.0,
    }

    print("==== [UnitTest] Construct agent ====")
    agent = PPOAgent(paras, hparams)

    # ---- 4) Sanity check: forward & sample once ----
    print("==== [UnitTest] Sanity: sample one action ====")
    state = _build_state(i=0, n=paras.n, prev_obj=0.0).to(agent.device)
    x_idx, y_vec, logp, value, ent = agent.sample_action(state)
    print("sampled x_idx:", int(x_idx))
    print("sampled y_vec shape:", tuple(y_vec.shape))
    print("logp/value/entropy:", float(logp), float(value), float(ent))

    # ---- 5) Sanity check: push one transition into buffer and update once ----
    print("==== [UnitTest] Sanity: buffer add & update once ====")
    X0, Y0 = _init_feasible_XY(paras)
    F_e = np.ones((paras.n, 1), dtype=np.float32) * (paras.f_e_max / paras.n)
    F_c = np.ones((paras.n, 1), dtype=np.float32) * (paras.f_c_max / paras.n)

    prev_obj = objective(X0, Y0, F_e, F_c, paras)
    X1, Y1 = agent._apply_action_to_XY(
        X0.copy(), Y0.copy(),
        user_i=0,
        x_idx=int(x_idx),
        y_vec=y_vec.detach().cpu().numpy().astype(np.float32),
    )
    new_obj = objective(X1, Y1, F_e, F_c, paras)
    reward = float(new_obj - prev_obj)

    # agent.buffer.clear()
    # agent.buffer.add(
    #     state.squeeze(0).detach().cpu(),
    #     int(x_idx),
    #     y_vec.detach().cpu(),
    #     logp.detach().cpu(),
    #     float(value),
    #     reward,
    #     1.0,  # done
    # )
    # agent.update_policy(epoch=0)
    agent.buffer.clear()

    # step 0
    agent.buffer.add(
        state.squeeze(0).detach().cpu(),
        int(x_idx),
        y_vec.detach().cpu(),
        logp.detach().cpu(),
        float(value),
        reward,
        0.0,  # done=0
    )

    # step 1: 再采样一次，构造第二条 transition
    state2 = _build_state(i=1, n=paras.n, prev_obj=new_obj).to(agent.device)
    x2, y2, logp2, v2, _ = agent.sample_action(state2)

    X2, Y2 = agent._apply_action_to_XY(
        X1.copy(), Y1.copy(),
        user_i=1,
        x_idx=int(x2),
        y_vec=y2.detach().cpu().numpy().astype(np.float32),
    )
    obj2 = objective(X2, Y2, F_e, F_c, paras)
    reward2 = float(obj2 - new_obj)

    agent.buffer.add(
        state2.squeeze(0).detach().cpu(),
        int(x2),
        y2.detach().cpu(),
        logp2.detach().cpu(),
        float(v2),
        reward2,
        1.0,  # done=1
    )

    agent.update_policy(epoch=0)
    print("update_policy() OK")

    print("update_policy() OK")

    # ---- 6) Full tiny train run ----
    print("==== [UnitTest] Run tiny train ====")
    best_val, best_sol, hist = agent.train()
    print("train() OK")
    print("best_val:", best_val)
    print("history:", hist)

