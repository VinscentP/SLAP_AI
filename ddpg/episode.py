from config import *
from robot_env import RobotEnv
from ddpg_agent import DDPGAgent
from tqdm import tqdm
import numpy as np

def episode():
    ddpg_agent = DDPGAgent()
    env = RobotEnv()

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
        episode_reward = 0
        current_state = env.reset()
        done = False

        for step in range(MAX_EP_STEPS):
            action = ddpg_agent.choose_action(current_state)
            
            #exploration noise 
            noise = np.random.normal(0, ACTION_NOISE_STD, size=ACTION_SPACE_SIZE)
            action = np.clip(action + noise, -1, 1)

            next_state, reward, done, _ = env.step(action)

            ddpg_agent.replay_buffer.add((current_state, action, reward, next_state, done))
            ddpg_agent.train()

            current_state = next_state
            episode_reward += reward

            if done:
                break