import pybullet_envs
import gym
import numpy as np
from brain import Agent
from utils import plot_learning_curve
from gym import wrappers
import os

# Continous action space - Where you have to perform a continous action instead of a discrete action
# Example of a continous action is steering a wheel, pressing a gas pedal where you also have to decide:
# how much to rotate the wheel and how much to press the gas pedal
# Actor Critic is used for continous action spaces

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 250
    filename = 'inverted_pendulum.png'
    figure_file = os.path.join('CartPole\SAC', filename)
    best_score = env.reward_range[0]
    score_history = []

    # make true to test the model and false for training
    load_checkpoint = True
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'episode {i+1}: ' 'score=%.2f' % score, 'average_score=%.2f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)