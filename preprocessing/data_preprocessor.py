from __future__ import annotations
import os
import glob
import json
import subprocess
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache

from totalsegmentator.python_api import totalsegmentator
from dipy.io.image import load_nifti
from monai.transforms import CropForeground

ARGS_PATH = 'arguments.json'
ORGANS = ['liver', 'spleen', 'kidney_right', 'kidney_left', 'small_bowel']

class Preprocessor:

    @lru_cache(maxsize=None)
    def get_args(self):
        """
        Loads the arguments from the JSON file and caches them.
        """
        with open(ARGS_PATH) as f:
            return json.load(f)
            
    def __init__(self):
        args = self.get_args()
        
        self.working_dir: str = args['working_dir']
        self.data_dir: str = args['data_folder_path']
        self.nifti_dir: str = args['data_niftis_path']
        os.makedirs(self.nifti_dir, exist_ok=True)
        self.segment_dir: str = args['data_segment_path']
        os.makedirs(self.segment_dir, exist_ok=True)

    def _get_dicom_subfolders(self) -> list:
        """
        Returns a list of unconverted DICOM subfolders.
        """
        dicom_subfolders = []
        prev_nifti_files = os.listdir(self.nifti_dir)

        for patient_folder in os.listdir(self.data_dir):
            patient_folder_path = os.path.join(self.data_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                for subfolder in os.listdir(patient_folder_path):
                    subfolder_path = os.path.join(patient_folder_path, subfolder)
                    nifti_folder_name = '_'.join(subfolder_path.split('/')[-2:]) + '.nii.gz'
                    if os.path.isdir(subfolder_path) and nifti_folder_name not in prev_nifti_files:
                        dicom_subfolders.append(subfolder_path)

        return dicom_subfolders

    def _cleanup_temp_files(self):
        """
        Cleans up temporary JSON files generated in the process.
        """
        json_files = glob.glob(f'{self.nifti_dir}/*.json')
        for file_path in json_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def _get_nifti_files_to_segment(self) -> list:
        """
        Returns a list of NIfTI files that are yet to be segmented.
        """
        prev_segment_files = os.listdir(self.segment_dir)
        nifti_files_to_segment = [
            os.path.join(self.nifti_dir, nifti_file)
            for nifti_file in os.listdir(self.nifti_dir)
            if nifti_file.split('/')[-1].replace('.nii.gz', '') not in prev_segment_files
        ]

        return nifti_files_to_segment

    def convert_dicom_to_nifti(self, dcm_folder_path: str) -> None:
        """
        Converts DICOM files to NIfTI using dcm2niix.

        Args:
            dcm_folder_path (str): Path to the folder containing DICOM files.
        """
        file_name = '_'.join(dcm_folder_path.split('/')[-2:]) + '.nii.gz'
        out_file_path = os.path.join(self.nifti_dir, file_name)

        cmd = f'dcm2niix -o {self.nifti_dir} -z y {dcm_folder_path}'
        subprocess.run(cmd, shell=True)

        generated_nifti = glob.glob(os.path.join(
            self.nifti_dir, '{}_*.nii.gz'.format(dcm_folder_path.split('/')[-1])))[0]

        os.rename(generated_nifti, out_file_path)

    def segment_nifti_files(self, nifti_file_path: str) -> None:
        """
        Executes TotalSegmentator on the generated NIfTI files.

        Args:
            nifti_file_path (str): Path to the folder containing NIfTI files.
        """
        out_path = os.path.join(self.segment_dir, nifti_file_path.split(
            '/')[-1].replace('.nii.gz', ''))
        totalsegmentator(nifti_file_path, out_path, roi_subset=ORGANS)

    def json_organs(self, subject) -> None:
        value = {subject: {}}

        transform = CropForeground(return_coords=True)

        for organ in ORGANS:
            mask, _ = load_nifti(os.path.join(self.segment_dir, subject, organ + ".nii.gz"))
            _, coords_in, coords_end = transform(np.expand_dims(mask, axis=0))

            coords_in = coords_in.tolist()
            coords_end = coords_end.tolist()

            value[subject][organ] = {
                'start': coords_in,
                'end': coords_end
            }

        # Check if the JSON file exists
        filename = os.path.join(self.working_dir, 'cropped_organs.json')
        if os.path.exists(filename):
            with open(filename, 'r') as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = {}

        # Update the existing data with the new subject's data
        existing_data.update(value)

        # Save the updated dictionary to JSON
        with open(filename, 'w') as json_file:
            json.dump(existing_data, json_file, indent=6)
    
    def crop_organs(self, subject):
        filename = os.path.join(self.working_dir, 'cropped_organs.json')
        with open(filename, 'r') as json_file:
            coordinates = json.load(json_file)

        os.makedirs(os.path.join(
            self.nifti_dir,
            subject.strip(".nii.gz")), exist_ok=True)
        
        img, _ = load_nifti(os.path.join(self.nifti_dir, subject))

        for organ in ORGANS:
            organ_coordinates = coordinates[subject.strip(".nii.gz")][organ]
            organ_img = img[
                organ_coordinates["start"][0]:organ_coordinates["end"][0],
                organ_coordinates["start"][1]:organ_coordinates["end"][1],
                organ_coordinates["start"][2]:organ_coordinates["end"][2]
            ]
            organ_img = nib.Nifti1Image(organ_img, np.eye(4))

            nib.save(organ_img, os.path.join(self.nifti_dir,
                                            subject.strip(".nii.gz"),
                                            organ + ".nii.gz"))
            
        os.rename(os.path.join(self.nifti_dir, 
                               subject), 
                  os.path.join(self.nifti_dir,
                               subject.strip(".nii.gz"),
                               subject))

    def __call__(self) -> None:
        NUM_THREADS = 3

        # Convert DICOM to NIfTI
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            executor.map(self.convert_dicom_to_nifti, self._get_dicom_subfolders()[:10])
        
        self._cleanup_temp_files()
        
        # Segment the NIfTI files
        with ProcessPoolExecutor(max_workers=3) as executor:
            executor.map(self.segment_nifti_files, self._get_nifti_files_to_segment())
        
        # Process JSON files and crop organs
        cropped_organs_json = os.listdir(self.segment_dir)
        cropped_organs = os.listdir(self.nifti_dir)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            executor.map(self.json_organs, cropped_organs_json)
            executor.map(self.crop_organs, cropped_organs)

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor()
