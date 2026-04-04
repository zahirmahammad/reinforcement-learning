import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import gymnasium as gym
import numpy as np
import os
import cv2
os.environ["MUJOCO_GL"] = "egl"

NUM_ENVS = 4
LR = 2e-4
TOTAL_TIMESTEPS = 1.5e7
NUM_STEPS = 100
DEVICE = "cpu"
GAMMA = 0.99
NUM_NET_UPDATES = 5
NUM_BATCHES = 3
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.02
LAMBDA = 0.95
MAX_GRAD_NORM = 0.5
TEST_FREQ = 1000


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
        entropy = normal_dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)



# Create training class
class Reacher_PPO:
    def __init__(self):
        self.vec_env = gym.vector.SyncVectorEnv([self.make_envs(i) for i in range(NUM_ENVS)])
        self.eval_env = gym.make("Reacher-v5", render_mode='rgb_array')

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
            env = gym.make("Reacher-v5")
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

    def TrainAgent(self):
        runs = int(TOTAL_TIMESTEPS // NUM_STEPS)
        obs = self.vec_env.reset()[0]
        obs = torch.from_numpy(obs).float().to(DEVICE)
        done = torch.zeros(NUM_ENVS).to(DEVICE)

        # Storage
        self.obs_arr = torch.zeros(NUM_STEPS, NUM_ENVS, self.state_dim[0]).to(DEVICE)
        self.actions_arr = torch.zeros(NUM_STEPS, NUM_ENVS, self.action_dim[0]).to(DEVICE)
        self.logprobs_arr = torch.zeros(NUM_STEPS, NUM_ENVS).to(DEVICE)
        self.rewards_arr = torch.zeros(NUM_STEPS, NUM_ENVS).to(DEVICE)
        self.values_arr = torch.zeros(NUM_STEPS, NUM_ENVS).to(DEVICE)
        self.dones_arr = torch.zeros(NUM_STEPS, NUM_ENVS).to(DEVICE)
        for run in range(runs):
            for step in range(NUM_STEPS):
                with torch.no_grad():
                    action, logprob, entropy, value = self.agent.get_action_value(obs)

                self.obs_arr[step] = obs
                self.dones_arr[step] = done
                self.actions_arr[step] = action
                self.logprobs_arr[step] = logprob
                self.values_arr[step] = value.flatten()

                obs, reward, terminated, truncated, _ = self.vec_env.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                done = torch.from_numpy(done).float().to(DEVICE)
                obs = torch.from_numpy(obs).float().to(DEVICE)
                
                self.rewards_arr[step] = torch.from_numpy(reward).float().to(DEVICE)

               
            # calculate returns
            advantages = torch.zeros_like(self.rewards_arr).to(DEVICE)
            lastgaelam = torch.zeros(NUM_ENVS).to(DEVICE)
            with torch.no_grad():
                last_value = self.agent.get_value(obs).reshape(1, -1)
                for r in reversed(range(NUM_STEPS)):
                    if r == NUM_STEPS - 1:
                        not_done = 1 - done
                        next_Vs = last_value
                    else:
                        not_done = 1 - self.dones_arr[r+1]
                        # next_Vs = returns[r+1]
                        next_Vs = self.values_arr[r+1]
                    delta = self.rewards_arr[r] + GAMMA * next_Vs * not_done - self.values_arr[r]
                    lastgaelam = delta + GAMMA * LAMBDA * not_done * lastgaelam
                    advantages[r] = lastgaelam
                    # returns[r] = self.rewards_arr[r] + (GAMMA * not_done * next_Vs)
                returns = advantages + self.values_arr
            
            # Create data batches for training
            d_obs = self.obs_arr.reshape(-1, self.state_dim[0])
            d_actions = self.actions_arr.reshape(-1, self.action_dim[0])
            d_log_probs = self.logprobs_arr.reshape(-1)
            d_returns = returns.reshape(-1)
            d_advantages = advantages.reshape(-1)
            d_advantage = (d_advantages - d_advantages.mean()) / (d_advantages.std() + 1e-8)

            # Do network updates for certain epochs
            data_size = NUM_ENVS * NUM_STEPS
            batch_size = data_size // NUM_BATCHES
            for update in range(NUM_NET_UPDATES):
                # pick randomly
                inds = np.arange(data_size)
                np.random.shuffle(inds)
                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    b_inds = inds[start:end]
                    _, log_prob, entropy, value = self.agent.get_action_value(d_obs[b_inds], d_actions[b_inds])

                    # ratio - curr policy / prev policy
                    log_ratio = log_prob - d_log_probs[b_inds]
                    ratio = torch.exp(log_ratio)

                    # normalize
                    advantage = d_advantage[b_inds]

                    objective_pg = torch.min(ratio * advantage, advantage * torch.clamp(ratio, (1 - CLIP_COEF), (1 + CLIP_COEF)))
                    loss_pg = -(objective_pg.mean())

                    loss_VF = 0.5 * ((value.flatten() - d_returns[b_inds]) ** 2).mean()

                    loss = loss_pg + VF_COEF * loss_VF - ENT_COEF*entropy.mean()

                    self.optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                    self.optim.step()
            print(f"Run {run}: Loss: {loss}")

            if run % TEST_FREQ == 0:
                self.TestAgent(num_eval_episodes=5)
            

    def TestAgent(self, num_eval_episodes):
        print("----------------------------------")
        print(f"Agent Performance Test")
        print("----------------------------------")   
        frames = []
        for i in range(num_eval_episodes):
            total_reward = 0
            obs, info = self.eval_env.reset()
            for _ in range(NUM_STEPS):
                obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0)
                with torch.no_grad():
                    mean = self.agent.actor_net(obs)
                    action = mean.flatten()
                obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy())
                total_reward += reward
                frame = self.eval_env.render()
                frame = (frame * 255).astype('uint8') if frame.dtype != 'uint8' else frame
                frame = np.ascontiguousarray(frame)
                cv2.putText(frame, f"Episode: {i}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255,255,255), 2, cv2.LINE_AA)
                frames.append(frame)
                if terminated or truncated:
                    obs, info = self.eval_env.reset()
                    break
            print(f"Eval Episode: {i}, Total Reward: {total_reward}")
        imageio.mimsave(f"media/06_ppo_Reacher.gif", frames, fps=30, loop=True)

if __name__ == "__main__":
    Reacher = Reacher_PPO()
    # Reacher.TakeRandomActions()
    Reacher.TrainAgent()
    Reacher.TestAgent(num_eval_episodes=10)