import os
import torch
from torch.utils.data import Dataset, DataLoader

LABEL_TO_INT = {
    "nature": 0, "ai": 1
}

class TinyGenImage(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []

        for type in os.listdir(data_dir):
            type_path = os.path.join(data_dir, type)
            label = LABEL_TO_INT.get(type)
            for image in os.listdir(type_path):
                file_path = os.path.join(type_path, image)
                self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        tensor = torch.load(file_path, weights_only=True)
        return tensor, label
