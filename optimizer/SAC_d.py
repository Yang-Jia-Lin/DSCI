import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Dict

from utils.device import get_device

State = np.ndarray
Action = Tuple[np.ndarray, np.ndarray]
Mask   = Dict[str, np.ndarray]
DEVICE = get_device()

class ReplayBuffer:
    def __init__(self, capacity: int, m: int, n_early: int):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.states  = np.zeros((capacity, 2), dtype=np.float32)  # [t/n, ones/2]
        self.x_logits = np.zeros((capacity, m), dtype=np.float32)
        self.y_vals   = np.zeros((capacity, n_early), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1), dtype=np.float32)
        self.next_idx = np.zeros(capacity, dtype=int)  # not used (terminal flag suffice)
        self.dones    = np.zeros((capacity, 1), dtype=bool)

    def push(self, s, x_logits, y_vals, r, done):
        self.states[self.ptr]  = s
        self.x_logits[self.ptr] = x_logits
        self.y_vals[self.ptr]   = y_vals
        self.rewards[self.ptr]  = r
        self.dones[self.ptr]    = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch: int):
        max_idx = self.capacity if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, size=batch)
        return (
            torch.tensor(self.states[idxs], device=DEVICE),
            torch.tensor(self.x_logits[idxs], device=DEVICE),
            torch.tensor(self.y_vals[idxs], device=DEVICE),
            torch.tensor(self.rewards[idxs], device=DEVICE),
            torch.tensor(self.dones[idxs], device=DEVICE),
        )

class Actor(nn.Module):
    def __init__(self, m: int, n_early: int):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc_x = nn.Linear(128, m)        # logits for X
        self.fc_y = nn.Linear(128, n_early)  # raw for Y(sigmoid)

    def forward(self, s: torch.Tensor):
        h = F.relu(self.fc1(s))
        return self.fc_x(h), torch.sigmoid(self.fc_y(h))


class Critic(nn.Module):
    def __init__(self, m: int, n_early: int):
        super().__init__()
        self.fc1 = nn.Linear(2 + m + n_early, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, s, x_logits, y):
        x_prob = torch.sigmoid(x_logits)  # treat logits as Bernoulli probs
        inp = torch.cat([s, x_prob, y], dim=-1)
        h = F.relu(self.fc1(inp))
        return self.fc2(h)


def _mask_logits(logits: torch.Tensor, x_mask: np.ndarray):
    mask_t = torch.tensor(x_mask, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(~mask_t, -1e9)  # 禁止位置给 -inf
    return logits


class SACDAgent:
    """简化离散 SAC，支持动作掩码 & ≤2 个 1 约束。"""
    def __init__(self, n_layers: int, early_layers: Sequence[int], buffer_cap=50_000, lr=0.0001):
        self.m = n_layers
        self.E = list(early_layers)
        self.nE = len(self.E)
        self.actor  = Actor(n_layers, self.nE).to(DEVICE)
        self.critic = Critic(n_layers, self.nE).to(DEVICE)
        self.critic_t = Critic(n_layers, self.nE).to(DEVICE)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_act = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_cri = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buf = ReplayBuffer(buffer_cap, n_layers, self.nE)
        self.gamma = 0.99
        self.tau = 0.005
        self.update_step = 0

    # ---------------- action 采样 -----------------
    def select_action(self, state: State, mask: Mask, deterministic: bool = False) -> Action:
        """根据 mask 采样动作；deterministic=True 时选取 top-k 概率最大动作。"""
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            x_logits, y_raw = self.actor(s)
        x_logits = x_logits.squeeze(0)
        y_raw    = y_raw.squeeze(0)
        logits_masked = _mask_logits(x_logits, mask["x_mask"])
        probs = torch.softmax(logits_masked, dim=-1).cpu().numpy()
        # 根据模式选取索引
        if deterministic:
            # 直接取概率最高的两个下标
            choose = np.argsort(probs)[-min(2, len(probs)):]
        else:
            # 随机采样，探索
            choose = np.random.choice(self.m, size=2, replace=False, p=probs)
        x_row = np.zeros(self.m, dtype=int)
        valid_choose = [idx for idx in choose if mask["x_mask"][idx]]
        x_row[valid_choose] = 1
        # Y 只在早退层：使用 sigmoid 输出并 clip
        y_row = np.ones(self.m)
        y_row[self.E] = y_raw.cpu().numpy()
        low, high = mask["y_low"][self.E], mask["y_high"][self.E]
        y_row[self.E] = np.clip(y_row[self.E], low, high)
        return x_row, y_row

    # ---------------- 经验存储 -----------------
    def store_transition(self, s, act, r, done):
        if s is None:  # 终止
            return
        x_logits, y_vals = self._pack_action(act)
        self.buf.push(s, x_logits, y_vals, r, done)

    def _pack_action(self, act: Action):
        x, y = act
        # logits 用 x 直接映射：1→+6, 0→‑6 近似 Bernoulli 参数
        x_logits = (x.astype(np.float32) * 12 - 6)  # 0→‑6,1→+6
        y_vals   = y[self.E].astype(np.float32)
        return x_logits, y_vals

    # ---------------- 更新 -----------------
    def update(self, batch_size=64):
        if self.buf.ptr < batch_size and not self.buf.full:
            return
        s, x_log, y_val, r, done = self.buf.sample(batch_size)
        with torch.no_grad():
            q_targ = self.critic_t(s, x_log, y_val)
            y_targ = r + (1 - done.float()) * self.gamma * q_targ
        q_pred = self.critic(s, x_log, y_val)
        loss_cri = F.mse_loss(q_pred, y_targ)
        self.opt_cri.zero_grad(); loss_cri.backward(); self.opt_cri.step()

        # Actor 更新：最大化 Q
        x_log_pi, y_pi = self.actor(s)
        loss_act = -self.critic(s, x_log_pi, torch.sigmoid(y_pi)).mean()
        self.opt_act.zero_grad(); loss_act.backward(); self.opt_act.step()

        # 软更新目标网络
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_t.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ---------------- 贪心评估 -----------------
    def eval_policy(self, n_users: int):
        xs, ys = [], []
        for t in range(n_users):
            state = np.array([t / n_users, 0.0], dtype=np.float32)
            mask = {"x_mask": np.ones(self.m, bool),
                    "y_low": np.zeros(self.m),
                    "y_high": np.ones(self.m)}
            x, y = self.select_action(state, mask, deterministic=True)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)
