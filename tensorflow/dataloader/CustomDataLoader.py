import numpy as np

class CustomDataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True, num_classes=None, input_size=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.X = X
        self.y = y
        
        # Automatically determine num_classes if not provided
        if num_classes is None:
            num_classes = np.max(y) + 1
        
        self.num_classes = num_classes
        self.num_samples = len(X)
        self.indices = np.arange(self.num_samples)
        self.curr_idx = 0
        
        if self.input_size is not None:
            self.X = np.reshape(self.X, (-1, self.input_size, 1))
        
        # Convert y to one-hot encoding
        if self.num_classes is not None:
            if np.max(self.y) >= self.num_classes:
                raise ValueError(f"Max label in y exceeds num_classes ({self.num_classes})")
            self.y = np.eye(self.num_classes)[self.y]
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_idx >= self.num_samples:
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.curr_idx = 0
            raise StopIteration
        
        batch_indices = self.indices[self.curr_idx:self.curr_idx + self.batch_size]
        self.curr_idx += self.batch_size
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        return X_batch, y_batch

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
