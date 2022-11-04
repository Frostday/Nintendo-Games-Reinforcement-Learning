import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    # evaluates the value of a state and action pair
    def __init__(self, beta, input_dims, n_actions, fc1_dims=32, fc2_dims=64,
                 name='critic', chkpt_dir='Mario/SAC/models'):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(input_dims[0], fc1_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc1_dims),
            nn.Conv2d(fc1_dims, fc2_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc2_dims),
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
        )
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q = self.critic(state)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    # estimates the value of a particular state, doesn't care about the action took or are taking
    def __init__(self, beta, input_dims, n_actions, fc1_dims=32, fc2_dims=64,
                 name='value', chkpt_dir='Mario/SAC/models'):
        super(ValueNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(input_dims[0], fc1_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc1_dims),
            nn.Conv2d(fc1_dims, fc2_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc2_dims),
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 1),
        )
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        v = self.critic(state)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    # returns a probability distribution
    def __init__(self, alpha, input_dims, fc1_dims=32,
                 fc2_dims=64, n_actions=2, name='actor', chkpt_dir='Mario/SAC/models'):
        # max_action will be multiplied to the probability distribution(b/w -1 and 1) to get the real range
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(input_dims[0], fc1_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc1_dims),
            nn.Conv2d(fc1_dims, fc2_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc2_dims),
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.Linear(512, 256)
        )
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.mu = nn.Linear(256, self.n_actions)
        # mean of probability distribution
        self.sigma = nn.Linear(256, self.n_actions)
        # standard deviation of probability distribution

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.actor(state)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        # clamp all values of sigma btw reparam_noise(almost 0) and 1

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        # to calculate the actual policy - required for continous action spaces
        # policy is a probability distribution that tells us probability of selecting any action in our action space given some state
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
            # just adds some extra noise
        else:
            actions = probabilities.sample()
        # print(actions)

        action = T.tanh(actions)

        log_probs = probabilities.log_prob(actions)
        # log of probabilities for loss function
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
