from torch.utils.data import Dataset
import torch
import numpy as np
from config import *

class CustomDatasetVal(Dataset):
    def __init__(self, df, data_transform=None):
        self.df = df
        self.data_transform = data_transform

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.df)

    def __getitem__(self, idx):
        cat_id = self.df.iloc[idx]["CATID"]

        # Get the target value for the given index
        values = torch.tensor(self.df.iloc[idx][2:len(self.df.columns)], dtype=torch.float32) # first two columns are cat_id and target
        
        # Apply the transformations if specified
        if self.data_transform:
            values = self.data_transform(values)

        # Get the target for the given index
        target = torch.tensor(self.df.iloc[idx][1], dtype=torch.long)

        return values, target, cat_id