import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPONetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.pi = nn.Linear(256, act_dim)
        self.fc3 = nn.Linear(512, 256)
        self.vf = nn.Linear(256, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        pi_logits = self.pi(h2)

        h3 = self.activation(self.fc3(h1))
        value = self.vf(h3).squeeze(-1)

        return pi_logits, value

    def act(self, obs, avail_actions):
        obs = torch.tensor(obs, dtype=torch.float32)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32)
        pi_logits, value = self.forward(obs)
        pi_logits = pi_logits + avail_actions  # Mask unavailable actions
        dist = torch.distributions.Categorical(logits=pi_logits)
        action = dist.sample()
        neglogp = -dist.log_prob(action)
        return action.numpy(), value.detach().numpy(), neglogp.detach().numpy()

    def evaluate_actions(self, obs, avail_actions, actions):
        pi_logits, value = self.forward(obs)
        pi_logits = pi_logits + avail_actions
        dist = torch.distributions.Categorical(logits=pi_logits)
        neglogp = -dist.log_prob(actions)
        entropy = dist.entropy()
        return neglogp, entropy, value

class PPOModel:
    def __init__(self, network, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, learning_rate=2.5e-4):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def train(self, lr, cliprange, observations, available_actions, returns, actions, values, neglogpacs):
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs = torch.tensor(observations, dtype=torch.float32)
        avail_actions = torch.tensor(available_actions, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        neglogpacs = torch.tensor(neglogpacs, dtype=torch.float32)
        advs = torch.tensor(advs, dtype=torch.float32)

        neglogpacs_new, entropy, values_pred = self.network.evaluate_actions(obs, avail_actions, actions)
        values_pred = values_pred.squeeze()

        ratio = torch.exp(neglogpacs - neglogpacs_new)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advs
        pg_loss = -torch.min(surr1, surr2).mean()

        value_clipped = values + (values_pred - values).clamp(-cliprange, cliprange)
        vf_loss1 = (values_pred - returns).pow(2)
        vf_loss2 = (value_clipped - returns).pow(2)
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return pg_loss.item(), vf_loss.item(), entropy_loss.item()
