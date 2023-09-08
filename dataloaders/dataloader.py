import os
import numpy as np
import pandas as pd
from glob import glob

from dipy.io.image import load_nifti

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, Resize, RandFlip, RandZoom, NormalizeIntensity

class RSNA_AbdominalInjuryDataset(Dataset):
    def __init__(self, img_path, df):
        self.img_path = img_path
        self.subjects = glob(os.path.join(
            self.img_path,
            "*"
        ))
        self.df = pd.read_csv(df)

        self.transform = Compose([
            Resize((112, 112, 112)),
            NormalizeIntensity(),
            RandZoom(prob=0.1, min_zoom=0.85, max_zoom=1.15)
        ])

    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        file_path = self.subjects[idx]

        if os.path.exists(file_path):
            img, _ = load_nifti(file_path)
            img = self.transform(
                np.expand_dims(img, axis=0)
            )
        else:
            print(f"File not found: {file_path}")

        target = self.df.loc[file_path.split("_")[0], :]

        return img, target


if __name__ == "__main__":
    dataset = RSNA_AbdominalInjuryDataset(
        img_path = "/home/fran/DATA/rsna-2023-abdominal-trauma-detection/nifti_output",
        df = "/home/fran/DATA/rsna-2023-abdominal-trauma-detection/train.csv"
    )
    dl = DataLoader(dataset, batch_size = 2, shuffle=True)