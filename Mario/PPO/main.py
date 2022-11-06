import numpy as np
import os
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

N = 200
batch_size = 64
n_epochs = 4
alpha = 0.00001

agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
                n_epochs=n_epochs, input_dims=env.observation_space.shape)
n_games = 300

filename = 'ppo_average_scores.png'
figure_file = os.path.join('Mario/PPO', filename)

best_score = env.reward_range[0]
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        env.render()
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    
    if avg_score > best_score:
        best_score = avg_score
        print('new best')

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score, 'learning_steps=%d' % learn_iters)

agent.save_models()
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, figure_file)
env.close()