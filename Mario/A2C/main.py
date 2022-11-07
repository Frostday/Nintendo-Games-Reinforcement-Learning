import sys
import torch
import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack

# hyperparameters
learning_rate = 3e-6

# Constants
GAMMA = 0.98
n_games = 300

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
        )


    def forward(self, state):
        value = self.critic(state)

        policy_dist = self.actor(state)

        return value, policy_dist


def a2c(env):
    num_inputs = env.observation_space.shape
    num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    score_history = []

    for episode in range(n_games):
        log_probs = []
        values = []
        rewards = []
        done = False
        score = 0

        state = env.reset()
        while not done:
            state_tensor = T.tensor(state.__array__(), dtype=T.float).unsqueeze(0)
            value, policy_dist = actor_critic.forward(state_tensor)
            # print(value, policy_dist)
            # print(value.shape, policy_dist.shape)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)
            score += reward

            if episode % 50 == 0:
                env.render()

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done:
                new_state_tensor = T.tensor(new_state.__array__(), dtype=T.float).unsqueeze(0)
                Qval, _ = actor_critic.forward(new_state_tensor)
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                sys.stdout.write("episode: {}, reward: {} \n".format(episode, np.sum(rewards)))
                break

        score_history.append(score)

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, "Mario/A2C/a2c_average_scores.png")


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    a2c(env)
