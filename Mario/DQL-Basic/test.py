import gym
from brain import DeepQNetwork
import torch as T

env = gym.make('LunarLander-v2')
DQN = DeepQNetwork(lr=0.001, n_actions=4, input_dims=[8], fc1_dims=256, fc2_dims=256)
DQN.load_state_dict(T.load('DQL-Basic/model.pt'))

score = 0
state = env.reset()
done = False

while not done:
    action = T.argmax(DQN(T.tensor([state]).to(DQN.device))).item()
    stateNew, reward, done, info = env.step(action)        
    env.render()
    score += reward
    state = stateNew

print('Score:', score)