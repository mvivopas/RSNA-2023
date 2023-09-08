from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess

from totalsegmentator.python_api import totalsegmentator

ARGS_PATH = 'arguments.json'


class Preprocessor:
    def __init__(self):
        """
        Initializes the Preprocessor class with the data folder
        path from arguments.json.
        """
        # Load arguments from a JSON file
        with open(ARGS_PATH) as f:
            args = json.load(f)

        self.data_dir: str = args['data_folder_path']

        self.nifti_dir: str = args['data_niftis_path']
        os.makedirs(self.nifti_dir, exist_ok=True)

        self.segment_dir: str = args['data_segment_path']
        os.makedirs(self.segment_dir, exist_ok=True)


    def __call__(self) -> None:
        """
        Main method to execute DICOM to NIfTI conversion and
        segmentation for all patients.
        """
        # TODO: MULTI-THREADING
        # Iterate over patient folders
        for patient_folder in os.listdir(self.data_dir):
            patient_folder_path = os.path.join(self.data_dir, patient_folder)

            # Check if it's a directory
            if os.path.isdir(patient_folder_path):
                # Iterate over subfolders containing DICOM files
                for subfolder in os.listdir(patient_folder_path):
                    subfolder_path = os.path.join(
                        patient_folder_path, subfolder,
                    )

                    # Check if it's a directory
                    if os.path.isdir(subfolder_path):
                        # Convert DICOM to NIfTI
                        self.convert_dicom_to_nifti(subfolder_path)

        # Remove all JSON files
        shutil.rmtree(glob.glob(f'{self.nifti_dir}/*.json'))

        for nifti_file in os.listdir(self.nifti_dir):
            nifti_file_path = os.path.join(self.nifti_dir, nifti_file)
            # Segment the NIfTI files
            self.segment_nifti_files(nifti_file_path)

    def convert_dicom_to_nifti(self, dcm_folder_path: str) -> None:
        """
        Converts DICOM files to NIfTI using dcm2niix.

        Args:
            dcm_folder_path (str): Path to the folder containing DICOM files.
        """
        file_name = '_'.join(dcm_folder_path.split('/')[-2:]) + ".nii.gz"
        out_file_path = os.path.join(self.nifti_dir, file_name)

        cmd = f'dcm2niix -o {self.nifti_dir} -z y {dcm_folder_path}'
        subprocess.run(cmd, shell=True)

        generated_nifti = glob.glob(os.path.join(self.nifti_dir, "{0}_*.nii.gz".format(dcm_folder_path.split("/")[-1])))[0]

        os.rename(generated_nifti,out_file_path)


    def segment_nifti_files(self, nifti_file_path: str) -> None:
        """
        Executes TotalSegmentator on the generated NIfTI files.

        Args:
            nifti_file_path (str): Path to the folder containing NIfTI files.
        """
        out_path = os.path.join(self.segment_dir, nifti_file_path.split('/')[-1].replace('.nii.gz',''))
        totalsegmentator(nifti_file_path, out_path)


if __name__ == '__main__':
    # Initialize the preprocessor
    preprocessor = Preprocessor()
    # Run the preprocessor for DICOM to NIfTI conversion and segmentation
    preprocessor()
