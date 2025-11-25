import torch
import pickle
from torch.utils.data import Dataset

class TCDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f: 
            self.X, self.y, self.sids = pickle.load(f) 
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        if self.y.dim() == 3: self.y = self.y.squeeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.sids[i] 