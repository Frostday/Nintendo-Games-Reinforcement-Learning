import gym
import numpy as np
import os
from brain import Agent
from utils import plot_learning_curve

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack
from gym.spaces import Box

import torch as T
import torchvision.transforms as Ts

env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = JoypadSpace(env, [["right"], ["right", "A"]])

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = T.tensor(observation.copy(), dtype=T.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = Ts.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = Ts.Compose(
            [Ts.Resize(self.shape), Ts.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

N = 200
batch_size = 5
n_epochs = 4
alpha = 0.00001
print(env.action_space.n)
print(env.observation_space.shape)
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, 
                n_epochs=n_epochs, input_dims=env.observation_space.shape)
n_games = 300

filename = 'mario.png'
figure_file = os.path.join('Mario\PPO', filename)

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