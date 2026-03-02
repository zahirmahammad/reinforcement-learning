import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class StateNetwork(nn.Module):
    def __init__(self, inp, outp, device):
        super(StateNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(inp, 16)
        self.fc2 = nn.Linear(16, outp)

    def forward(self, x):
        self.x = F.relu(self.fc1(x))
        self.x = self.fc2(x)
        self.x = F.softmax(self.x, dim=1)
        return self.x

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        return action, m.log_prob(action)
