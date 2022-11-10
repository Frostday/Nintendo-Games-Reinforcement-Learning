from utils import plot_learning_curve
import numpy as np
import os
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])

n_games = 10

filename = 'Mario/Human/human_scores.png'
scores = []
best_score = env.reward_range[0]

iters = 0
for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        # choosing action (assign A=["right"], S=["right", "A"])
        action =
        observation_, reward, done, info = env.step(action)
        env.render()
        score += reward
        observation = observation_
    scores.append(score)
    avg_score = np.mean(scores[-100:])

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score)

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, filename)
