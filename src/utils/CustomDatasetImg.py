from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
import sys
sys.path.append("..")
from config import *

class CustomDatasetImg(Dataset):
    def __init__(self, df, data_transform=None, data_dir = IMG_DIR):
        self.df = df
        self.data_dir = data_dir
        self.data_transform = data_transform

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.df)

    def __getitem__(self, idx):
        cat_id = self.df.iloc[idx]["CATID"]
        img_path = f"{self.data_dir}/{cat_id}_rgb.png"
        image = np.array(read_image(img_path)).transpose(1, 2, 0)
        
        # Apply the transformations if specified
        if self.data_transform:
            image = self.data_transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Get the target value for the given index
        target = torch.tensor(self.df.iloc[idx][1], dtype=torch.long)

        return image, target, cat_id