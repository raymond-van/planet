import random

# Experience Replay Buffer
class ExpReplay():
    def __init__(self):
        self.replay = []
        self.batch_sz = 50
        self.chunk_len = 50
        
    # Sample batch of sequences
    def sample_batch(self):
        batch = []
        for i in range(self.batch_sz):
            idx = random.randint(0, len(self.replay)-self.chunk_len)
            batch.append((self.replay[idx:idx+self.chunk_len]))
        return batch
