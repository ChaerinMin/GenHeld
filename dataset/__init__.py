from .base_dataset import _P3DFaces
from .base_dataset import HandData, ObjectData
from torch.utils.data import Dataset
import torch

class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
        return
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return torch.zeros(1,1)