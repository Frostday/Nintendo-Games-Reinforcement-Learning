from utils import plot_learning_curve
import numpy as np
import os
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from gym.utils import play
import pygame
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])

pygame.init()
pygame.key.set_repeat(10, 0)
display = pygame.display.set_mode((300, 300))
n_games = 10
# play.play(env, zoom=3)

filename = 'Mario/Human/human_scores.png'
scores = []
best_score = env.reward_range[0]

iters = 0
for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    action = 0
    while not done:
        # choosing action (assign default=["right"], space bar=["right", "A"])
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action = 0

        observation_, reward, done, info = env.step(action)
        env.render()
        score += reward
        observation = observation_
    scores.append(score)
    avg_score = np.mean(scores[-100:])

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score)

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, filename)
env.close()
