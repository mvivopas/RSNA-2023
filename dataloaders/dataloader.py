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

ORGANS_DICT = {'liver': 'liver', 
               'spleen': 'spleen', 
               'kidney': 'kidney_right', # TODO: Adapt to both kidneys 
               'bowel': 'small_bowel'}

class RSNA_AbdominalInjuryDataset(Dataset):

    """
    Custom Dataset for loading RSNA Abdominal Injury data.

    This class facilitates loading of the NIFTI format images, applying 
    transformations to them, and obtaining the corresponding labels from a CSV file.

    Attributes:
    - target_organ (str): The target organ of interest.
    - sessions (list): List of file paths to the data.
    - label_df (pd.DataFrame): DataFrame containing the labels for each patient.
    - transforms (Compose): A MONAI Compose object containing image transformations.

    Parameters:
    - data_path (str): Path to the directory containing the NIFTI data.
    - csv_path (str): Path to the CSV file containing labels.
    - target_organ (str): The target organ name that we are interested in.
    """

    def __init__(self, data_path, csv_path, target_organ):
        self.target_organ = target_organ
        self.sessions = glob(f'{data_path}/*')
        self.label_df = pd.read_csv(csv_path).set_index('patient_id')
        
        # Mapping the column names to corresponding indices
        column_mapping = {i: idx for idx, i in enumerate(self.label_df.filter(like=self.target_organ, axis=1).columns.to_list())}
        
        # Modifying label_df to have categorical mapping
        self.label_df = self.label_df.filter(like=self.target_organ, axis=1).idxmax(axis=1).map(column_mapping)

        # Defining image transformations to apply
        self.transforms = Compose([
            Resize((112, 112, 112)),  # Resizing the image to shape (112, 112, 112)
            NormalizeIntensity(),     # Normalizing image intensity
        ])

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.sessions[idx], ORGANS_DICT[self.target_organ] + ".nii.gz")
        img, _ = load_nifti(img_path)

        # Applying transformations to the image
        img = self.transforms(np.expand_dims(img, axis=0))

        # Extracting the subject's name from the file path and retrieving its label
        subject_name = img_path.split('/')[-2].split('_')[0]
        label = self.label_df.loc[int(subject_name),]

        return img.float(), label