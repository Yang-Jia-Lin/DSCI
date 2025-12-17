# 存储 rollout 的缓冲区

import torch


class RolloutBuffer:
    def __init__(self):
        self.dones = None
        self.rewards = None
        self.values = None
        self.logprobs = None
        self.actions_Y = None
        self.actions_X = None
        self.states = None
        self.clear()

    def clear(self):
        self.states = []
        self.actions_X = []
        self.actions_Y = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, state, action_X, action_Y, logprob, value, reward, done):
        self.states.append(state)
        self.actions_X.append(action_X)
        self.actions_Y.append(action_Y)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_advantages(self, gamma, lam):
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def get_all_data(self):
        return {
            'states': self.states.copy(),
            'actions_X': self.actions_X.copy(),
            'actions_Y': self.actions_Y.copy(),
            'logprobs': self.logprobs.copy(),
            'values': self.values.copy(),
            'rewards': self.rewards.copy(),
            'dones': self.dones.copy()
        }

    def extend(self, data):
        self.states += data['states']
        self.actions_X += data['actions_X']
        self.actions_Y += data['actions_Y']
        self.logprobs += data['logprobs']
        self.values += data['values']
        self.rewards += data['rewards']
        self.dones += data['dones']


class TopKRolloutMemory:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buffer = []

    def add(self, rollout, reward):
        self.buffer.append((rollout, reward))
        self.buffer.sort(key=lambda x: x[1], reverse=True)  # 按 reward 排序
        if len(self.buffer) > self.capacity:
            self.buffer.pop()

    def get_all(self):
        return [r[0] for r in self.buffer]
