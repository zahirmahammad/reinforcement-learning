import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import gymnasium as gym
import numpy as np

NUM_ENVS = 4
LR = 0.0002

# Create Agent
class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Agent, self).__init__()
        self.actor_net = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, action_dim), std=0.01)
        )
        
        self.actor_net_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_net = nn.Sequential(
            self.layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0)
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic_net(x)

    def get_action_value(self, x, action=None):
        mean = self.actor_net(x)
        log_std = self.actor_net_logstd.expand_as(mean)
        std = torch.exp(log_std)

        # Normal dist
        normal_dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = normal_dist.sample()

        log_prob = normal_dist.log_prob(action).sum(-1)
        entropy = normal_dist.entropy(action).sum(-1)
        return action, log_prob, entropy, self.get_value(x)



# Create training class
class Pusher_PPO:
    def __init__(self):
        self.vec_env = gym.vector.SyncVectorEnv([self.make_envs(i) for i in range(NUM_ENVS)])
        self.eval_env = gym.make("Pusher-v5")

        self.state_dim = self.vec_env.single_observation_space.shape
        self.action_dim = self.vec_env.single_action_space.shape
        
        print("---------------------------------------------") 
        print(f"Observation Space Dim: {self.state_dim}")
        print(f"Action Space Dim: {self.action_dim}")
        print("---------------------------------------------") 

        self.agent = Agent(self.state_dim[0], self.action_dim[0])
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=LR)

    def make_envs(self, seed):
        def callable_env():
            env = gym.make("Pusher-v5")
            env.reset(seed=seed)
            env.observation_space.seed(seed=seed)
            env.action_space.seed(seed=seed)
            return env
        return callable_env

    def TakeRandomActions(self):
        self.vec_env.reset()
        for i in range(10):
            action = self.vec_env.action_space.sample()
            obs, reward, terminated, truncated, info = self.vec_env.step(action)
            print(f"Action Taken: {action}, Current Reward: {reward}")
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                print("Episode Terminated!!")
                self.vec_env.reset()

    def TraingAgent(self):
        pass

    def TestAgent(self):
        pass


if __name__ == "__main__":
    pusher = Pusher_PPO()
    pusher.TakeRandomActions()