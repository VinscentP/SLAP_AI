import time
from deep_q_learning.deep_q import DQNAgent
from deep_q import DQNAgent
from robot_env import RobotEnv
import numpy as np
import random
from tqdm import tqdm

#loss function and optimizer
#training loop episodes
#epsilon should start fairly low ~0.3 and decay at a faster rate, but must never reach 0 in order to keep exploring

#epsilon_t = max(epsilon_min, epsilon_min + (epsilon_init - epsilon_min) * math.exp(-decay * time_step))

EPISODES = 1000
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
epsilon = 1.0

ep_rewards = []

def episode_loop():
    agent = DQNAgent()
    env = RobotEnv() 

    current_state = env.rest()
    
    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episodes'):
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            
            new_state, reward, done = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

            ep_rewards.append(episode_reward)

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)