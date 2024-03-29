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

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=batchsize, n_actions=env.action_space.n,
                eps_end=0.05, input_dims=env.observation_space.shape, lr=lr)

score_history, eps_history = [], []
best_score = env.reward_range[0]
figure_file = os.path.join('Mario/DQL-Basic', "dql_basic_average_scores.png")

# print(env.observation_space.shape)
# T.autograd.set_detect_anomaly(True)
iters = 0
for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if i % 20 == 0:
            env.render()
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        if iters % 20 == 0:
            agent.learn()
        iters = (iters + 1) % 20
        observation = observation_
    score_history.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(score_history[-100:])

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score, 'epsilon=%.2f' % agent.epsilon)

T.save(agent.Q_eval.state_dict(), 'Mario/DQL-Basic/model.pt')

x = [i+1 for i in range(n_games)]
plot_learning_curve(x, score_history, figure_file)
env.close()