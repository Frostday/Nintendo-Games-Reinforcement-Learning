from utils import plot_learning_curve
import numpy as np
import os
import torch as T
from brain import Agent
from utils import *
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

lr = 0.00001
batchsize = 64
n_games = 300
testing_mode = False

agent = Agent(gamma=0.99, epsilon=1.0, alpha=lr, input_dims=env.observation_space.shape, 
                n_actions=env.action_space.n, mem_size=10000, eps_min=0.01, 
                batch_size=batchsize, eps_dec=1e-3, replace=100)

if testing_mode:
    agent.load_models()
    
filename = 'Mario/DQL-Dueling/ddql_average_scores.png'
scores, eps_history = [], []
best_score = env.reward_range[0]

iters = 0
for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if i % 50 == 49:
            env.render()
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        if not testing_mode and iters % 20 == 0:
            agent.learn()
        iters = (iters + 1) % 20
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])

    if avg_score > best_score:
            best_score = avg_score
            if not testing_mode:
                agent.save_models()

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score, 'epsilon=%.2f' % agent.epsilon)

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, filename)