import random

# Experience Replay Buffer
class ExpReplay():
    def __init__(self, batch_sz, chunk_len):
        self.replay = []
        self.batch_sz = batch_sz
        self.chunk_len = chunk_len
        
    # Sample batch of sequences
    def sample_batch(self):
        batch = []
        for i in range(self.batch_sz):
            idx = random.randint(0, len(self.replay)-self.chunk_len)
            batch.append((self.replay[idx:idx+self.chunk_len]))
        return batch
