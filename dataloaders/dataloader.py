import os
import numpy as np
import pandas as pd
from glob import glob

from dipy.io.image import load_nifti

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, Resize, RandFlip, RandZoom, NormalizeIntensity

class RSNA_AbdominalInjuryDataset(Dataset):
    def __init__(self, data_path, csv_path):
        self.img_path = data_path
        self.sessions = os.listdir(self.img_path)
        self.csv = pd.read_csv(csv_path).set_index("patient_id")

        self.transforms = Compose([
            Resize((112, 112, 112)),
            NormalizeIntensity()
        ])

    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        img_name = self.sessions[idx]
        subject_name = img_name.split("_")[0]

        img, _ = load_nifti(os.path.join(self.img_path, img_name))
        img = self.transforms(
            np.expand_dims(img, axis=0)
        )
        label = self.csv.loc[int(subject_name), "bowel_healthy"]
    
        return (img.float(), int(label))
