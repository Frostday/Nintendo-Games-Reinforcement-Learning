import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.cap = batch_size * 20
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.probs = []  # log_probs calculated by actor
        self.vals = []  # values calculated by critic

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (np.array(self.states), np.array(self.actions),
                np.array(self.probs), np.array(self.vals),
                np.array(self.rewards), np.array(self.dones), batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

        if len(self.states) > self.cap:
            self.states.pop(0)
            self.actions.pop(0)
            self.probs.pop(0)
            self.vals.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.probs = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=32, fc2_dims=64, chkpt_dir='Mario/PPO/models/'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor')
        self.actor = nn.Sequential(
            nn.Conv2d(input_dims[0], fc1_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc1_dims),
            nn.Conv2d(fc1_dims, fc2_dims, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(fc2_dims),
            nn.Flatten(),
            nn.Linear(20736, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        # print(dist)
        dist = Categorical(dist)
        # print(dist.probs)
        # get categorical distribution from probabilities
        return dist 

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=32, fc2_dims=64, chkpt_dir='Mario/PPO/models/'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic')
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

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation.__array__(), dtype=T.float).unsqueeze(0).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        # print(action)

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, val_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = val_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # to calculate advantage
            # a_t gives us the advantage of state t(also considering future states)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k]))-values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            # training
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # can also do - prob_ratio = (new_probs - old_probs).exp()
                # we take exponential of log probabilities to get the real probabilities
                weighted_probs = prob_ratio*advantage[batch]
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor.zero_grad()
                self.critic.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # clear memory after all epochs are done
        self.memory.clear_memory()
