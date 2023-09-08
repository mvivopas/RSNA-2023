#!/usr/bin/env python3

import os
import subprocess
from totalsegmentator.python_api import totalsegmentator
from concurrent.futures import ThreadPoolExecutor

def dicom_to_nifti(dcm2niix, input_path, output_base_path, verbose=False):
    """
    Convert DICOM files to NIFTI using dcm2niix.
    """
    
    # Extract patient and study directory names from the input path
    study_dir, patient_dir = os.path.basename(input_path), os.path.basename(os.path.dirname(input_path))

    # Construct the output directory path
    output_dir = os.path.join(output_base_path, patient_dir, study_dir, "NIFTI")
    os.makedirs(output_dir, exist_ok=True)

    # Construct the command for dcm2niix
    verbose_str = "-v 2" if verbose else "-v 0"
    command = f"\"{dcm2niix}\" -o \"{output_dir}\" -z y {verbose_str} \"{input_path}\""
    
    # Execute the command
    subprocess.call(command, shell=True)
    
    return output_dir

def process_session(session_path, nifti_root):
    patient_name = os.path.basename(os.path.dirname(session_path))
    session_name = os.path.basename(session_path)
    
    # Convert DICOMs to NIFTI
    nifti_output_folder = os.path.join(nifti_root, patient_name, session_name, 'NIFTI')
    os.makedirs(nifti_output_folder, exist_ok=True)
    
    dcm2niix_path = "/home/fran/SOFTWARE/fsl/bin/dcm2niix"  # Add the path to dcm2niix here
    nifti_path = dicom_to_nifti(dcm2niix_path, session_path, nifti_output_folder)

    # Segment the NIFTI
    segmentations_folder = os.path.join(nifti_root, patient_name, session_name, 'Segmentations')
    os.makedirs(segmentations_folder, exist_ok=True)
    totalsegmentator(nifti_path, segmentations_folder)

def process_patient_folder(root_folder, nifti_root, max_threads=4):
    session_paths = []
    for patient in os.listdir(root_folder):
        patient_path = os.path.join(root_folder, patient)
        for session in os.listdir(patient_path):
            session_path = os.path.join(patient_path, session)
            session_paths.append(session_path)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(lambda path: process_session(path, nifti_root), session_paths)

if __name__ == '__main__':
    dicom_root = "/home/fran/DATA/rsna-2023-abdominal-trauma-detection/train_images"
    nifti_root = "/home/fran/DATA/rsna-2023-abdominal-trauma-detection/nifti_files"
    process_patient_folder(dicom_root, nifti_root)
