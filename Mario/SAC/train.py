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

batch_size = 64
alpha = 0.00001

agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                input_dims=env.observation_space.shape)
n_games = 2
filename = 'sac_average_scores.png'
figure_file = os.path.join('Mario/SAC', filename)

if __name__ == '__main__':
    best_score = env.reward_range[0]
    score_history = []

    # make true to test the model and false for training
    testing_mode = False
    if testing_mode:
        agent.load_models()
        env.render(mode='human')
    iters = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if i % 50 == 0:
                env.render()
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not testing_mode and iters % 20 == 0:
                agent.learn()
            iters = (iters + 1) % 20
            observation = observation_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not testing_mode:
                agent.save_models()

        print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score)

    if not testing_mode:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)