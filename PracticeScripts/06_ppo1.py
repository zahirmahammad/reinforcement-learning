import torch
import torch.nn as nn
import imageio
import numpy as np
import gymnasium as gym


NUM_ENVS = 6
NUM_STEPS = 150
TOTAL_TIMESTEPS = 1e6
GAMMA = 0.99
LR = 2.5e-4
DEVICE = 'cpu'
NUM_BATCHES = 4     # depends on NUM_STEPS
UPDATE_EPOCHS = 5
CLIP_COEF = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()

        self.actor_net = nn.Sequential(
            self.layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, action_dim), std=0.01)
        )

        # logstd param : log(standard_deviation)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic_net = nn.Sequential(
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
        return self.critic_net(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_net(x)
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)

        normal_dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = normal_dist.sample()

        # “How likely is this entire action vector?” - thats why we add the log
        # not “How likely is each dimension separately?”
        log_prob = normal_dist.log_prob(action).sum(-1) 
        entropy = normal_dist.entropy().sum(-1)
        return action, log_prob, entropy, self.get_value(x)


class PPO_BiPedWalker:
    def __init__(self):
        self.eval_env = gym.make("BipedalWalker-v3", render_mode='rgb_array')
    
        self.vec_env = gym.vector.SyncVectorEnv([self.make_envs(i) for i in range(NUM_ENVS)])
        self.state_dim = self.vec_env.single_observation_space.shape
        self.action_dim = self.vec_env.single_action_space.shape

        print("---------------------------------------------")
        print(f"Observation Space Dim: {self.state_dim}")
        print(f"Action Space Dim: {self.action_dim}")
        print("---------------------------------------------")

        self.agent = Agent(self.state_dim[0], self.action_dim[0])
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=LR)

        # Storage
        self.obs_arr = torch.zeros(NUM_STEPS, NUM_ENVS, self.state_dim[0])
        self.actions_arr = torch.zeros(NUM_STEPS, NUM_ENVS, self.action_dim[0])
        self.rewards_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.dones_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.logprobs_arr = torch.zeros(NUM_STEPS, NUM_ENVS)
        self.values_arr = torch.zeros(NUM_STEPS, NUM_ENVS)

    def make_envs(self, seed):
        def callable_env():
            env = gym.make("BipedalWalker-v3")
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
            print(f"Action taken:{action}, Current Reward: {reward}")
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                print("Episode Terminated!!")
                self.vec_env.reset() 
        
    
    def TrainAgent(self):
        epochs = int(TOTAL_TIMESTEPS // NUM_STEPS)
        obs = torch.tensor(self.vec_env.reset()[0]).to(DEVICE)
        done = torch.zeros(NUM_ENVS).to(DEVICE)
        for ep in range(epochs):
            for step in range(NUM_STEPS):
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.get_action_and_value(obs)

                self.obs_arr[step] = obs
                self.dones_arr[step] = done
                obs, reward, terminated, truncated, info = self.vec_env.step(action)
                obs = torch.tensor(obs).to(DEVICE)
                done = np.logical_or(terminated, truncated)
                done = [int(i) for i in done]
                done = torch.tensor(done).to(DEVICE)
                self.actions_arr[step] = action
                self.logprobs_arr[step] = log_prob
                self.values_arr[step] = value.flatten()
                self.rewards_arr[step] = torch.tensor(reward).to(DEVICE)

            # calculate returns
            returns = torch.zeros_like(self.rewards_arr).to(DEVICE)
            with torch.no_grad():
                last_value = self.agent.get_value(obs).reshape(1, -1)
                for r in reversed(range(NUM_STEPS)):
                    if r == NUM_STEPS - 1:
                        next_not_done = 1.0 - done
                        next_value_state = last_value
                    else:
                        next_not_done = 1.0 - self.dones_arr[r+1]
                        next_value_state = returns[r + 1]
                    returns[r] = self.rewards_arr[r] + GAMMA * next_not_done * next_value_state
                advantages = returns - self.values_arr

            d_obs = self.obs_arr.reshape(-1, self.state_dim[0])
            d_actions = self.actions_arr.reshape(-1, self.action_dim[0])
            d_log_probs = self.logprobs_arr.reshape(-1)
            d_values = self.values_arr.reshape(-1)
            d_returns = returns.reshape(-1)
            d_advantages = advantages.reshape(-1)

            # Update
            data_size = NUM_STEPS * NUM_ENVS
            batch_size = data_size // NUM_BATCHES
            for e in range(UPDATE_EPOCHS):
                inds = np.arange(data_size)
                np.random.shuffle(inds)
                for start in range(0, data_size, batch_size):
                    end = start + batch_size
                    b_inds = inds[start:end]
                    _, log_prob, entropy, values = self.agent.get_action_and_value(d_obs[b_inds], d_actions[b_inds])

                    # ratio
                    log_ratio = log_prob - d_log_probs[b_inds]
                    ratio = torch.exp(log_ratio)

                    advantage = (d_advantages[b_inds] - d_advantages[b_inds].mean()) / (d_advantages[b_inds].std() + 1e-8)

                    objective_pg = torch.min(ratio * advantage, advantage* torch.clamp(ratio, (1 - CLIP_COEF), (1 + CLIP_COEF)))
                    loss_pg = -(objective_pg.mean())

                    loss_VF = 0.5 * ((values.view(-1) - d_returns[b_inds]) ** 2).mean()

                    loss = loss_pg + VF_COEF*loss_VF - ENT_COEF*entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()

            print(f"Epoch: {ep} ; Reward: {torch.sum(self.rewards_arr, axis=0)}; Loss: {loss}")

 
    def testAgent(self, num_eval_episodes):
        print("----------------------------------")
        print(f"Agent Performance Test")
        print("----------------------------------")   
        obs, info = self.eval_env.reset()
        frames = []
        for i in range(num_eval_episodes):
            total_reward = 0
            for _ in range(NUM_STEPS):
                obs = torch.tensor(obs).to(DEVICE).unsqueeze(0)
                action, log_prob, ent, value = self.agent.get_action_and_value(obs)
                action = action.flatten()
                obs, reward, terminated, truncated, info = self.eval_env.step(action.cpu().numpy())
                total_reward += reward
                frame = self.eval_env.render()
                frames.append(frame)
                if terminated or truncated:
                    obs, info = self.eval_env.reset()
                    break
            print(f"Eval Episode: {i}, Total Reward: {total_reward}")
        imageio.mimsave(f"media/06_ppo_BiPedalWalker.gif", frames, fps=30, loop=True)


            




if __name__ == "__main__":
    print("---------------------------------------------")
    print("PPO Implementation (Continuous Action Space)")
    print("Env : BipedalWalker-v3")
    print("---------------------------------------------")
        
    ppo = PPO_BiPedWalker()
    # ppo.TakeRandomActions()
    ppo.TrainAgent()
    ppo.testAgent(num_eval_episodes=10)