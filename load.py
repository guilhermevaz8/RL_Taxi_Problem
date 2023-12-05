import gymnasium as gym
from stable_baselines3 import A2C, PPO
import numpy

models_dir = "models/A2C"

env = gym.make('Taxi-v3', render_mode="human")  
env.reset()

model_path = f"{models_dir}/10000.zip"
model = A2C.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, trunc, done, info = env.step(action)
        env.render()
        print(ep, rewards, done)
        print("---------------")