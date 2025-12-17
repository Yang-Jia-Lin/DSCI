# ActorCritic 网络

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim_X, action_dim_Y):
        super(ActorCritic, self).__init__()

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Actor X 输出 Bernoulli logits
        self.actor_X = nn.Linear(128, action_dim_X)

        # Actor Y 输出高斯均值
        self.actor_Y_mu = nn.Linear(128, action_dim_Y)
        self.actor_Y_logstd = nn.Parameter(torch.zeros(action_dim_Y) - 2)

        # Critic 输出值函数
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)

        logits_X = self.actor_X(features)
        mu_Y = self.actor_Y_mu(features)
        std_Y = self.actor_Y_logstd.exp()
        value = self.critic(features)

        return logits_X, mu_Y, std_Y, value

