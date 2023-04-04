from collections import deque
import random

class ReplayMemory:
    def __init__(self,memLength):
        self.memory=deque(maxlen=memLength)
    
    def push(self,transition):
        self.memory.append(transition)
    
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
