import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split

class CustomImageDataset(Dataset):
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        example = torch.IntTensor(self.examples[idx])
        label = self.labels[idx]

        return example, label