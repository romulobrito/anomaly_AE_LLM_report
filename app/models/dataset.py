from torch.utils.data import Dataset
import torch

class PavimentoDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.dataset[index]) 