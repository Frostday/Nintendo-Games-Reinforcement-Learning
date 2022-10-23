import gym
from brain import Agent
from utils import plot_learning_curve
import numpy as np
import os
import torch as T

os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym.make('LunarLander-v2')
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_end=0.05, input_dims=[8], lr=0.001)
scores, eps_history = [], []
n_games = 500

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        # if i % 10 == 0:
        #     env.render()
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-100:])

    print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score, 'epsilon=%.2f' % agent.epsilon)

    T.save(agent.Q_eval.state_dict(), 'DQL-Basic/model.pt')

x = [i+1 for i in range(n_games)]
filename = 'DQL-Basic/lunar_lander.png'
plot_learning_curve(x, scores, eps_history, filename)