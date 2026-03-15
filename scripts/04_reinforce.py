import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import imageio

class PolicyNetwork(nn.Module):
    def __init__(self, ninput, noutput, device):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(ninput, 16)
        self.fc2 = nn.Linear(16, noutput)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probabs = self.forward(state)
        m = Categorical(probabs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class MCReinforce:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name, render_mode='rgb_array')
 
    def ShowEnvInfo(self):
        print(f"Environment Information: {self.env_name}")
        print(f"---------------------------------------------------")
        print(f"Observation Space Dimension: {self.env.observation_space.shape}")
        print(f"Action Space Dimension: {self.env.action_space.n}")
        print(f"---------------------------------------------------")

    def TakeRandomActions(self):
        obs, info = self.env.reset()
        for i in range(10):
            action = self.env.action_space.sample()
            print(f"Step {i+1} : Action Taken: {action}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
    
    def ReinforceAlgo(self, n_episodes, max_steps, gamma, lr, device):
        self.gamma = gamma
        self.lr = lr

        self.policy = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n, device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        score_deque = deque(maxlen=100)
        scores = []

        for i in range(1, n_episodes+1):
            saved_log_probs = []
            rewards = []

            state, _ = self.env.reset()

            for s in range(max_steps):
                action, log_prob = self.policy.act(state)
                saved_log_probs.append(log_prob)

                state, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break

            score_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            # Compute discounted return for each state, action
            returns = deque(maxlen=max_steps)
            n_steps = len(rewards)

            for t in reversed(range(n_steps)):
                disc_return_t = returns[0] if len(returns)>0 else 0
                returns.appendleft(rewards[t] + self.gamma*disc_return_t)

            eps = np.finfo(np.float32).eps.item()   # smallest number
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # loss
            policy_loss = []
            for log_prob_i, disc_return_i in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob_i * disc_return_i)
            
            policy_loss = torch.stack(policy_loss).sum()

            # update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if i % 50 == 0:
                print(f"Episode: {i} \t Average Score: {np.mean(score_deque):.2f}")

            if max(score_deque)>0 and len(score_deque)>1 and len(set(score_deque)) == 1:  # checks if all the elements in score_deque is same
                print("Max reward acheived!!")
                print("Stopping training....")
                break
            
        return self.policy, scores


    def EvaluatePolicy(self, num_episodes, max_steps):
        self.eval_env = gym.make(self.env_name, render_mode='rgb_array')
        ep_rewards = []
        frames = []
        for n in range(num_episodes):
            obs, info = self.eval_env.reset()
            total_reward = 0
            for i in range(max_steps):
                action, log_prob = self.policy.act(obs)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                frame = self.eval_env.render()
                frames.append(frame)
                if terminated or truncated:
                    break
            ep_rewards.append(total_reward)
            print(f"Episode: {n} \t Reward: {total_reward}")
        mean_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        self.eval_env.close()
        
        imageio.mimsave(f"media/{self.env_name}.gif", frames, fps=30, loop=True)
        return mean_reward, std_reward


# ---- Practice Environments ----
# Image based state
class MCReinforce2:
    def __init__(self):
        pass


class CNNPolicyNetwork(nn.Module):
    def __init__(self, input_ch, output_actions):
        super(CNNPolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        # self.fc1 = nn.Linear(64 *)




if __name__ == "__main__":
    # reinforce = MCReinforce("CartPole-v1")
    # reinforce.ShowEnvInfo()
    # reinforce.TakeRandomActions()
    # policy, scores = reinforce.ReinforceAlgo(n_episodes=8000, max_steps=1000, gamma=0.99, lr=1e-2, device='cpu')
    # reinforce.EvaluatePolicy(num_episodes=10, max_steps=1000)

    reinforce = MCReinforce("LunarLander-v3")
    reinforce.ShowEnvInfo()
    reinforce.TakeRandomActions()
    policy, scores = reinforce.ReinforceAlgo(n_episodes=12000, max_steps=100, gamma=0.99, lr=1e-2, device='cpu')
    reinforce.EvaluatePolicy(num_episodes=10, max_steps=100)

    # reinforce = MCReinforce("Acrobot-v1")
    # reinforce.ShowEnvInfo()
    # reinforce.TakeRandomActions()
    # reinforce.ReinforceAlgo(n_episodes=10000, max_steps=1000, gamma=0.99, lr=1e-2, device='cpu')
    # reinforce.EvaluatePolicy(num_episodes=10, max_steps=1000)