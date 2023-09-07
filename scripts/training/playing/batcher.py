import numpy as np

class Batcher(object):
    
    def __init__(self, n, batch_size):
        assert(n >= batch_size)
        self.n = n
        self.batch_size = batch_size
        self.n_batches = n // batch_size
        self.batch_i = 0
        self.indexes = np.arange(n)
        self.shuffled_indexes = np.random.permutation(self.indexes)
        
    def next_batch(self, arrays):
        result = []
        
        for a in arrays:
            start_i = self.batch_i * self.batch_size
            end_i = start_i + self.batch_size
            batch_indexes = self.shuffled_indexes[start_i : end_i]
            result.append(a[batch_indexes])
        
        self.batch_i += 1
        if self.batch_i >= self.n_batches:
            self.batch_i = 0
            self.shuffled_indexes = np.random.permutation(self.indexes)
            
        return tuple(result)
