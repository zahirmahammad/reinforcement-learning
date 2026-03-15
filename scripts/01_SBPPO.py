import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

import imageio
from IPython.display import Video, display

'''
PPO, stablebaselines3
'''
class Unit1Practice:
  def __init__(self):
    self.env = gym.make("LunarLander-v3")
    self.eval_env = gym.make("LunarLander-v3", render_mode = 'rgb_array')
    # self.eval_env = RecordVideo(self.eval_env, video_folder="eval_videos", episode_trigger=lambda episode_id: True, name_prefix="eval")
    self.eval_env = Monitor(self.eval_env)


  def ShowEnvInfo(self):
    obs, info = self.env.reset()
    print("------------------------------")
    print(f"Observation Space Dimension: {self.env.observation_space.shape}")
    print(f"Action Space Dimension: {self.env.action_space.n}")
    print("------------------------------")


  def TrainRLAgent(self):
    self.model = PPO('MlpPolicy', self.env, verbose=1)
    self.model = PPO('MlpPolicy',
                     env = self.env,
                     n_steps = 1024,
                     batch_size = 32,
                     n_epochs = 4,
                     gamma = 0.99,
                     gae_lambda = 0.98,
                     ent_coef = 0.01,
                     verbose = 1
                     )
    self.model.learn(total_timesteps=2e6)
    self.model.save("mylunarlander")



  def EvalModel(self, model_path, num_eval_episodes):
    # mean_reward, std_reward = evaluate_policy(model, self.eval_env, n_eval_episodes=10, deterministic = True)
    # print(f"mean_reward: {mean_reward} +/- {std_reward}")
    model = PPO.load(model_path, env=self.eval_env)
    obs, info = self.eval_env.reset()
    li_ep_rewards = []
    frames = []
    for i in range(num_eval_episodes):
      print(f"Evaluation: {i}")
      ep_reward = 0
      for i in range(1000):
        action, _ = model.predict(obs, deterministic = True)
        obs, rewards, terminated, truncated, info = self.eval_env.step(action)
        ep_reward += rewards

        frame = self.eval_env.render()  # get rgb array
        frames.append(frame)
        if terminated or truncated:
          obs, info = self.eval_env.reset()
          break
      li_ep_rewards.append(ep_reward)
    print(f"Mean reward : {sum(li_ep_rewards)/len(li_ep_rewards)}")
    imageio.mimsave("evals_video.gif", frames, fps=30)


  def VectorizeEnv(self, n_envs):
    # Independent method, doesnot work with others in class
    self.env = make_vec_env("LunarLander-v3", n_envs=n_envs)


  def TestEnvwithRandomActions(self):
    for _ in range(20):
      action = self.env.action_space.sample()
      print(f"Action Taken: {action}")
      obs, reward, terminated, truncated, info = self.env.step(action)
      if terminated or truncated:
        print("####--- Env is Reset ---###")
        obs, info = self.env.reset()
    self.env.close()


if __name__ == "__main__":
  unit1 = Unit1Practice()
  unit1.ShowEnvInfo()
  unit1.TrainRLAgent()
  unit1.EvalModel("mylunarlander", num_eval_episodes=10)