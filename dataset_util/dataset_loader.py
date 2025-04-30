import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

LABEL_TO_INT = {
    "nature": 0, "ai": 1
}

class SampledGenImage(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []

        # convert each image in folder to labelled data entry for correct ground truth labels
        for data_class in os.listdir(data_dir):
            type_path = os.path.join(data_dir, data_class)
            label = LABEL_TO_INT.get(data_class)
            for image in os.listdir(type_path):
                file_path = os.path.join(type_path, image)
                self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        tensor = torch.load(file_path, map_location='cpu')
        return tensor, label

