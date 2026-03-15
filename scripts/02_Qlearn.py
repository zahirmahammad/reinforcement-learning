import numpy as np
import gymnasium as gym
import imageio
from tqdm import tqdm
'''
Q Learning
'''
class Unit2_1:
    def __init__(self):
        self.env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="rgb_array")
        _, _ = self.env.reset()
        self.test_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="rgb_array")

    def PrintEnvInfo(self):
        print("------------------------------------------")
        print(f"Frozen Lake Env")
        print("------------------------------------------")
        print(f"Observation Space Dimension: {self.env.observation_space}")
        print(f"Action Space Dimension: {self.env.action_space.n}")
        print("------------------------------------------")


    def TrainAgent(self):
        state_dim = self.env.observation_space.n
        action_dim = self.env.action_space.n

        Qtable = np.zeros((state_dim, action_dim))

        def greedy_policy(Qtable, state):
            action = np.argmax(Qtable[state, :])
            return action

        def epsilon_greedy_policy(Qtable, state, epsilon):
            random_num = np.random.uniform(0, 1)
            if random_num < epsilon:
                action = self.env.action_space.sample()
            else:
                action = greedy_policy(Qtable, state)
            return action
        
        # Hyperparameters
        n_episodes = 100000
        lr = 0.7
        
        n_eval_ep = 10

        max_steps = 99
        gamma = 0.95

        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.0005

        # Train
        for ep in tqdm(range(n_episodes)):
            state, info = self.env.reset()
            terminated = False
            truncated = False
            epsilon = min_epsilon + ((max_epsilon - min_epsilon)*np.exp(-decay_rate*ep))
            # print(f"Epsilon : {epsilon}")
            for i in range(max_steps):
                action = epsilon_greedy_policy(Qtable, state, epsilon)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                # test without the brackets also
                Qtable[state, action] = Qtable[state, action] + lr*((reward + (gamma*Qtable[new_state, greedy_policy(Qtable, new_state)])) - Qtable[state, action])
                if terminated or truncated:
                    break
                state = new_state
        return Qtable

    def TestAgent(self, Qtable, num_eval_episodes):
        img_frames = []
        for ep in range(num_eval_episodes):
            state, info = self.test_env.reset()
            reward = 0
            for _ in range(100):
                action = np.argmax(Qtable[state, :])
                state, reward, terminated, truncated, info = self.test_env.step(action)
                reward += reward
                frame = self.test_env.render()
                img_frames.append(frame)
                if terminated or truncated:
                    print("------------------------------------")
                    print(f"Test {ep} -> Reward: {reward}")
                    print("------------------------------------")
                    break
        imageio.mimsave("unit2_1_evals.mp4", img_frames, fps=5)
        

    def TakeRandomActions(self):
        obs, info = self.env.reset()
        reward = 0
        for i in range(10):
            action = self.env.action_space.sample()
            print(f"{i}: Action taken : {action}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward += reward
            if terminated or truncated:
                print("--------------------------")
                print("Terminated!!")
                print(f"Total Reward = {reward}")
                print("--------------------------")
                reward = 0
                obs, info = self.env.reset()
            

class Unit2_2:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode="rgb_array")
        _, _ = self.env.reset()
        self.eval_env = gym.make("Taxi-v3", render_mode="rgb_array")

    def TrainAgent(self):
        state_dim = self.env.observation_space.n
        action_dim = self.env.action_space.n

        Qtable = np.zeros((state_dim, action_dim))

        # Parameters
        n_episodes = 10_000
        max_steps = 100
        gamma = 0.5
        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.005
        lr = 0.5

        def epsilon_greedy_policy(Qtable, state, epsilon):
            random_num = np.random.uniform(0, 1)
            if random_num < epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(Qtable[state, :])
            return action

        # train
        for ep in tqdm(range(n_episodes)):
            state, _ = self.env.reset()
            epsilon = min_epsilon + (np.exp(-decay_rate*ep)*(max_epsilon-min_epsilon))
            for i in range(max_steps):
                action = epsilon_greedy_policy(Qtable, state, epsilon)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                Qtable[state, action] = Qtable[state, action] + lr*(reward + gamma*np.max(Qtable[new_state, :]) - Qtable[state, action])
                state = new_state
                if terminated or truncated:
                    break
                    
        return Qtable



    def TestAgent(self, Qtable, num_eval_episodes):
        img_frames = []
        for ep in range(num_eval_episodes):
            state, _ = self.eval_env.reset()
            reward = 0
            for i in range(100):
                action = np.argmax(Qtable[state, :])
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                reward += reward
                frame = self.eval_env.render()
                img_frames.append(frame)
                if terminated or truncated:
                    print("----------------------------------")
                    print(f"Test {ep} -> Reward = {reward}")
                    print("----------------------------------")
                    reward = 0
                    break
        imageio.mimsave("unit2_2_evals.mp4", img_frames, fps=5)

    def PrintEnvInfo(self):
        print("-------------------------------------------------")
        print("Taxi Env")
        print("-------------------------------------------------")
        print(f"Observation Space dimension: {self.env.observation_space.n}")
        print(f"Action Space Dimension: {self.env.action_space.n}")
        print("-------------------------------------------------")

    def TakeRandomActions(self):
        pass


if __name__ == "__main__":
    unit2_1 = Unit2_1()
    unit2_1.PrintEnvInfo()
    # unit2_1.TakeRandomActions()
    Qtable = unit2_1.TrainAgent()
    unit2_1.TestAgent(Qtable, 10)

    unit2_2 = Unit2_2()
    unit2_2.PrintEnvInfo()
    Qtable = unit2_2.TrainAgent()
    unit2_2.TestAgent(Qtable, 10)