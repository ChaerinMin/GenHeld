import torch
from torch.utils.data import Dataset

from .base_dataclass import PaddedTensor, HandData, ObjectData, SelectorData, SelectorTestData, _P3DFaces


class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length
        return
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return torch.zeros(1,1)