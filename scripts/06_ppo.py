import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import imageio
import gymnasium as gym
import argparse


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, action_dim), std=0.01)
        )

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(state_dim, 64)),
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
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        prob = Categorical(logits=logits)
        if action is None:
            action = prob.sample()            
        return action, prob.log_prob(action), prob.entropy(), self.critic(x)


NUM_ENVS = 4
LR = 2.5e-4
GAMMA = 0.99
DEVICE = 'cpu'

NUM_STEPS = 128
NUM_TIMESTEPS = 1.8e6
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
GAE = 0

def make_env(seed):
    def callable_env():
        env = gym.make("LunarLander-v3")
        env.reset(seed=seed)        # same initial state everytime
        # Optionally seed action and observation spaces
        env.action_space.seed(seed)     # same action 
        env.observation_space.seed(seed)    # same obs
        return env
    return callable_env

class MyPPO:
    def __init__(self):
        # initialize env
        self.eval_env = gym.make("LunarLander-v3", render_mode='rgb_array')

        # Vector of envs (multiple envs)
        self.vec_env = gym.vector.SyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])
        self.state_dim = self.vec_env.single_observation_space.shape[0]
        self.action_dim = self.vec_env.single_action_space.n
        print(f"Observation Space: {self.state_dim}")
        print(f"Action Space: {self.action_dim}")

        self.agent = Agent(self.state_dim, self.action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=LR)

        # Storage
        self.obs_arr = torch.zeros(NUM_STEPS, NUM_ENVS, self.state_dim)
        self.actions_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.rewards_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.dones_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.logprobs_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.values_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        # -------



    def TakeRandomActions(self):
        print("---------------------------------------------")
        print("Testing random Actions")
        obs, _ = self.vec_env.reset()
        print(obs.shape)
        for i in range(10):
            action = self.vec_env.action_space.sample()
            obs, reward, terminated, truncated, info = self.vec_env.step(action)
            print(f"Step: {i}, Action: {action}, Reward: {reward}")
            # if terminated or truncated:
                # print("Env terminated, Resetting..")
                # obs, info = self.env.reset()
        print("---------------------------------------------")


    def train(self):
        num_updates = int(NUM_TIMESTEPS // (NUM_STEPS * NUM_ENVS))
        obs = torch.tensor(self.vec_env.reset()[0]).to(DEVICE)
        done = torch.zeros(NUM_ENVS).to(DEVICE)
        for update in range(num_updates):
            for step in range(NUM_STEPS):
                self.obs_arr[step] = obs
                self.dones_arr[step] = done
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(obs)
                self.actions_arr[step] = action
                self.logprobs_arr[step] = log_prob
                self.values_arr[step] = value.flatten()

                obs, reward, terminated, truncated, info = self.vec_env.step(action.cpu().numpy())
                self.rewards_arr[step] = torch.tensor(reward).to(DEVICE)
                obs = torch.tensor(obs).to(DEVICE)
                done = np.logical_or(terminated, truncated)
                done = [int(i) for i in done]
                done = torch.tensor(done).to(DEVICE)

            # Compute returns and advantages of the episode
            with torch.no_grad():
                last_value = self.agent.get_value(obs).reshape(1, -1)
                if GAE:
                    advantages = torch.zeros_like(self.rewards_arr).to(DEVICE)
                    # no idea what that is
                    pass
                else:
                    returns = torch.zeros_like(self.rewards_arr).to(DEVICE)
                    for t in reversed(range(NUM_STEPS)):
                        if t == NUM_STEPS - 1:
                            not_done = 1 - done
                            v_next_state = last_value
                        else:
                            not_done = 1 - self.dones_arr[t + 1]
                            v_next_state = returns[t + 1]
                        returns[t] = self.rewards_arr[t] + GAMMA * not_done * v_next_state
                    advantages = returns - self.values_arr

            # ----- Flatten Stuff ------
            b_obs = self.obs_arr.reshape(-1, self.state_dim)
            b_logprobs = self.logprobs_arr.reshape(-1)
            b_actions = self.actions_arr.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values_arr.reshape(-1)


            # ----- Update the Parameters ----
            batch_size = NUM_STEPS * NUM_ENVS
            minibatch_size = batch_size // NUM_MINIBATCHES
            for epoch in range(UPDATE_EPOCHS):
                inds = np.arange(batch_size)
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = inds[start:end]
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (b_advantages[mb_inds].std() + 1e-8)
                    pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)).mean()

                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()
            
            print(f"Update: {update} done: Reward: {torch.sum(self.rewards_arr, axis=0)}: Loss: {loss}")



    def testAgent(self, num_episodes):
        print("----------------------------------")
        print(f"Agent Performance Test")
        print("----------------------------------")    
        obs, info = self.eval_env.reset()
        frames = []
        for i in range(num_episodes):
            tot_reward = 0
            while True:
                obs = torch.tensor(obs).to(DEVICE)
                action, logprob, ent, value = self.agent.get_action_and_value(obs)
                obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy())
                tot_reward += reward
                frame = self.eval_env.render()
                frames.append(frame)
                if terminated or truncated:
                    obs, info = self.eval_env.reset()
                    break
            print(f"Eval Episode: {i}, Total Reward: {tot_reward}")
        imageio.mimsave(f"media/06_ppo.gif", frames, fps=30, loop=True)

    




if __name__ == "__main__":
    print(f"------------------------------------------------")
    print("PPO (Proximal Policy Optimization) Implementation")
    print(f"------------------------------------------------")

    ppo = MyPPO()
    ppo.TakeRandomActions()
    ppo.train()
    ppo.testAgent(num_episodes=10)