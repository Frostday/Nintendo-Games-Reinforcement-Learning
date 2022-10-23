import gym
from brain import Agent
import torch as T

env = gym.make('LunarLander-v2')
agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4, input_dims=[8], 
                n_actions=4, mem_size=1000000, eps_min=0.01, 
                batch_size=64, eps_dec=1e-3, replace=100)
agent.load_models()

score = 0
state = env.reset()
done = False

while not done:
    action = agent.choose_action(state)
    stateNew, reward, done, info = env.step(action)        
    env.render()
    score += reward
    state = stateNew

print('Score:', score)