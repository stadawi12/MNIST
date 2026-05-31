import torch
from torch.utils.data import Dataset

from main import MAX_PIXEL_VALUE

print(MAX_PIXEL_VALUE)

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].float() / MAX_PIXEL_VALUE  # Normalize the image
        label = self.labels[idx]
        return image, label