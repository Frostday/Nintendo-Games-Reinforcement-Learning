import gym
from brain import Agent
from utils import plot_learning_curve
import numpy as np
import os
import torch as T

os.environ['KMP_DUPLICATE_LIB_OK']='True'
env = gym.make('LunarLander-v2')
load_checkpoint = False

agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4, input_dims=[8], 
                n_actions=4, mem_size=1000000, eps_min=0.01, 
                batch_size=64, eps_dec=1e-3, replace=100)

if load_checkpoint:
    agent.load_models()
    
filename = 'LunarLander/DQL-Dueling/lunar_lander.png'
scores, eps_history = [], []
n_games = 500
best_score = env.reward_range[0]

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if i % 100 == 0:
            env.render()
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])

    if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score, 'epsilon=%.2f' % agent.epsilon)

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, eps_history, filename)