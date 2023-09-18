from __future__ import annotations

import os
from glob import glob

import numpy as np
import pandas as pd
from dipy.io.image import load_nifti
from monai.transforms import Compose
from monai.transforms import NormalizeIntensity
from monai.transforms import Resize
from torch.utils.data import Dataset


class RSNA_AbdominalInjuryDataset(Dataset):
    def __init__(self, data_path, csv_path, target_organ):
        self.target_organ = target_organ
        self.sessions = glob(f'{data_path}/*')
        self.label_df = pd.read_csv(csv_path).set_index('patient_id')

        self.label_names = self.label_df.filter(
            like=self.target_organ,
            axis=1,
        ).columns.to_list()

        self.transforms = Compose([
            Resize((112, 112, 112)),
            NormalizeIntensity(),
        ])

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        img_path = self.sessions[idx]
        img, _ = load_nifti(img_path)

        img = self.transforms(
            np.expand_dims(img, axis=0),
        )

        subject_name = os.path.basename(img_path).split('_')[0]
        label_values = self.label_df.filter(
            like=self.target_organ, axis=1,
        ).loc[int(subject_name),]

        return (img.float(), label_values)
