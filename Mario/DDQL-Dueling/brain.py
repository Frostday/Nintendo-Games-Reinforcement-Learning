import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir='LunarLander/DDQL-Dueling/models'):
        super(DuelingDeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)

        self.fc1 = nn.Linear(*input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(device=self.device)

    def forward(self, state):
        f1 = F.relu(self.fc1(state))
        V = self.V(f1)
        A = self.A(f1)

        return V, A

    def save_checkpoint(self):
        print('... Saving Checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... Loading Checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='LunarLander/DDQL-Dueling/models'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(
            self.lr, self.n_actions, name='q_eval', input_dims=self.input_dims, chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(
            self.lr, self.n_actions, name='q_next', input_dims=self.input_dims, chkpt_dir=self.chkpt_dir)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(
                self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)
        # array from 0 to batch_size-1

        V_s, A_s = self.q_eval.forward(states)
        # current state by q_eval
        V_s_, A_s_ = self.q_next.forward(states_)
        # next state by q_next
        V_s_eval, A_s_eval = self.q_eval.forward(states_)
        # next state by q_eval

        # predicted q values from V and A
        # we subtract mean to solve the identifiability issue
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        # q_pred is calculated only for action taken
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        # q_next is calculated for all actions after our current action has been taken
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        # q_eval is also calculated for all actions after our current action has been taken

        # the best actions of the next state according to the evaluation network
        max_actions = T.argmax(q_eval, dim=1)
        # where next state is terminal state
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        # we use value of the action given by target network but action is recommended by the evaluation network

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
