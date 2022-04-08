import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience Replay Buffer
class ExpReplay():
    def __init__(self, batch_sz, chunk_len, action_dim, mem_sz=10000):
        self.idx = 0
        self.obs = torch.empty((mem_sz, 3, 64, 64))
        self.actions = torch.empty((mem_sz, action_dim))
        self.rewards = torch.empty((mem_sz, ))
        self.batch_sz = batch_sz
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.mem_sz = mem_sz
        self.full = False
                                        
    # Sample batch of sequences
    def sample_batch(self):
        obs_batch = torch.empty((self.batch_sz, self.chunk_len, 3, 64, 64))
        action_batch = torch.empty((self.batch_sz, self.chunk_len, self.action_dim))
        reward_batch = torch.empty((self.batch_sz, self.chunk_len, ))
        for i in range(self.batch_sz):
            if self.full:
                idx = random.randint(0, self.mem_sz-self.chunk_len)
            else:
                idx = random.randint(0, self.idx-self.chunk_len)
            obs_batch[i] = self.obs[idx:idx+self.chunk_len]
            action_batch[i] = self.actions[idx:idx+self.chunk_len]
            reward_batch[i] = self.rewards[idx:idx+self.chunk_len]
        return obs_batch.to(device), action_batch.to(device), reward_batch.to(device)
    
        
    def append(self, frame, action, reward):
        self.obs[self.idx] = frame
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        if (self.idx + 1) == self.mem_sz: self.full = True
        self.idx = (self.idx + 1) % self.mem_sz
