# RSNA-2023: Traumatic Injury Diagnosis and Severity Grading

## Introduction and Overview

### Contextual Background

Traumatic injury constitutes a critical and pervasive public health issue on a global scale. These injuries result in over 5 million deaths annually worldwide, rendering accurate and early diagnosis a crucial component for effective medical intervention. Computed Tomography (CT) scans have become an essential tool for evaluating suspected abdominal injuries by providing detailed cross-sectional imagery.

### Problem Statement

The complexity and time-consuming nature of interpreting CT scans for abdominal trauma, especially in cases with multiple injuries or subtle areas of active bleeding, pose a challenge for healthcare professionals. These limitations can compromise the immediacy and effectiveness of medical interventions, thus impacting patient outcomes adversely.

### Objective of This Repository

Aligned with the goals of the Kaggle competition, this repository seeks to utilize Artificial Intelligence (AI) and Machine Learning (ML) for the rapid and accurate detection and severity grading of abdominal injuries based on CT scans. The intention is to develop advanced algorithms that can revolutionize trauma care standards globally.

---
## Preprocessing.py: DICOM to NIFTI Conversion Script

*What Does This Script Do?*

The `preprocessing.py` script automates the conversion of medical images from DICOM (Digital Imaging and Communications in Medicine) format to NIFTI (Neuroimaging Informatics Technology Initiative) format. Simply put, this script takes a collection of 2D images (each as a DICOM file) that together make up a 3D scan of a patient's body and converts them into a single 3D file (NIFTI format) for easier analysis and manipulation.
Why Convert from DICOM to NIFTI?

1. **Ease of Use**: DICOM files are essentially individual 2D slices with valuable metadata but cumbersome for 3D analysis. NIFTI consolidates these slices into one 3D image, making it easier to work with.

2. **Standardization**: NIFTI is a commonly used format in medical imaging research. It is supported by a plethora of image processing libraries and software.

3. **Compatibility**: Many image analysis tools and algorithms prefer or require NIFTI format, ensuring our dataset's broader applicability.

4. **Efficiency**: Managing a single 3D file is often more efficient, both in terms of computational resources and workflow, compared to working with multiple 2D slices.

Advantages of NIFTI for Deep Learning

1. **Batch Processing**: When using deep learning frameworks, it's often more efficient to pass in batches of 3D volumes. NIFTI files enable this, leading to faster training times.

2. **Data Augmentation**: NIFTI format simplifies the implementation of 3D data augmentation techniques, which are crucial for improving the model's performance and robustness.

3. **3D Context**: Storing the entire volume in a single file allows algorithms to easily leverage the 3D context, which is often important for medical imaging tasks.

4. **Memory Efficient**: NIFTI files can be more memory-efficient than a large number of individual 2D DICOM files, making it easier to manage large datasets during deep learning model training.

How Does preprocessing.py Work?

1. **Folder Structure**: The script assumes a main folder called train_images containing individual folders for each patient. Each patient's folder has one or more folders for scanning sessions, housing the actual DICOM files.

```plaintext
train_images/
├── Patient_1/
│   └── Session_1/
│       ├── file1.dcm
│       ├── file2.dcm
│       └── ...
├── Patient_2/
│   └── Session_1/
│       ├── file1.dcm
│       ├── file2.dcm
│       └── ...
```

2. **Conversion**: For each patient and scanning session, the script reads all DICOM files, organizes them into the correct sequence, and then converts them into a single NIFTI file.

3. **Output**: The converted NIFTI files are saved in an output folder, nifti_output, while maintaining the original folder structure.

```plaintext
nifti_output/
├── Patient_1/
│   └── Session_1.nii.gz
├── Patient_2/
│   └── Session_1.nii.gz
```

*How to execute the script?*

First enter the script and change the path routes to your owns (*train_images_folder* and *nifti_output_folder*). Then execute the following:

`python preprocessing.py`