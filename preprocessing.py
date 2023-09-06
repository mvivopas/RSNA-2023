#!/usr/bin/env python

import os
import pydicom
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_dicom_files(folder_path):
    """
    Load all DICOM files from a specified folder, sorted by their InstanceNumber.
    
    Parameters:
    folder_path (str): The path to the folder containing the DICOM files.
    
    Returns:
    list: A sorted list of pydicom Dataset objects.
    """
    dicom_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.dcm'):
            dicom_file_path = os.path.join(folder_path, filename)
            dicom_file = pydicom.dcmread(dicom_file_path)
            dicom_files.append(dicom_file)
    dicom_files.sort(key=lambda x: int(x.InstanceNumber))
    return dicom_files

def dicom_to_nifti(dicom_files, output_path):
    """
    Convert a list of DICOM files to a single NIFTI file.
    
    Parameters:
    dicom_files (list): A list of pydicom Dataset objects, each representing a DICOM file.
    output_path (str): The path where the NIFTI file will be saved.
    """
    img_shape = list(dicom_files[0].pixel_array.shape)
    img_shape.append(len(dicom_files))
    img_array = np.zeros(img_shape, dtype=np.int16)
    for i, dicom_file in enumerate(dicom_files):
        img_array[:, :, i] = dicom_file.pixel_array
    nifti_img = nib.Nifti1Image(img_array, np.eye(4))
    nib.save(nifti_img, output_path)

def load_and_convert_dicom_series(patient_path, nifti_output_base):
    """
    Load and convert all DICOM series for a patient to NIFTI format.
    
    Parameters:
    patient_path (str): The path to the patient's folder containing DICOM series folders.
    nifti_output_base (str): The base output directory where the NIFTI files will be stored.
    """
    patient_id = os.path.basename(patient_path)
    nifti_patient_folder = os.path.join(nifti_output_base, patient_id)

    if not os.path.exists(nifti_patient_folder):
        os.makedirs(nifti_patient_folder)

    for session in os.listdir(patient_path):
        session_path = os.path.join(patient_path, session)
        if os.path.isdir(session_path):
            dicom_files = load_dicom_files(session_path)
            nifti_output_path = os.path.join(nifti_patient_folder, f"{session}.nii.gz")
            dicom_to_nifti(dicom_files, nifti_output_path)

if __name__ == "__main__":
    # Base folder containing all the patients' folders
    train_images_folder = "/home/fran/DATA/rsna-2023-abdominal-trauma-detection/train_images"

    # Output folder for NIFTI files
    nifti_output_folder = "/home/fran/DATA/rsna-2023-abdominal-trauma-detection/nifti_output"

    # Create the output folder if it does not exist
    if not os.path.exists(nifti_output_folder):
        os.makedirs(nifti_output_folder)

    # List all patient folders
    patient_folders = [os.path.join(train_images_folder, patient) for patient in os.listdir(train_images_folder) if os.path.isdir(os.path.join(train_images_folder, patient))]

    # Parallel conversion
    with ThreadPoolExecutor() as executor:
        executor.map(load_and_convert_dicom_series, patient_folders, [nifti_output_folder]*len(patient_folders))

    print(f"Converted DICOM files in {train_images_folder} to NIFTI format at {nifti_output_folder}")