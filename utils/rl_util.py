import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity, batch_size) -> None:
        self.buffer = collections.deque(maxlen = capacity)
        self.batch_size = batch_size

    def add(self, state, action, reward):
        state = state.unbind(dim=0)
        action = action.unbind(dim=0)
        reward = reward.unbind(dim=0)

        self.buffer.extend(zip(state, action, reward))
    
    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        state, action, reward = zip(*transitions)
        state, action, reward = torch.stack(state, dim=0), torch.stack(action, dim=0), torch.stack(reward, dim=0)
        return state, action, reward
    
    def size(self):
        return len(self.buffer)
    
