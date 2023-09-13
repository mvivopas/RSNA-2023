# RSNA-2023: Traumatic Injury Diagnosis and Severity Grading

![License](https://img.shields.io/badge/license-MIT-blue.svg)

#### Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In line with the objectives of the [Kaggle competition][kaggle-url], this GitHub repository is dedicated to providing a streamlined, precise, and high-performing solution for detecting and grading the severity of abdominal injuries using CT scans. Our primary aim is to craft an image preprocessing pipeline that impeccably aligns with the needs of a model. This model, when supplied with a set of organ segmentations, will estimate the probability of the organs' illness/health. Thus, enhancing diagnostic capabilities in the realm of abdominal injuries.


## Installation

To set up the Jurisprudence Semantic Search tool on your local machine, follow these steps:

1. Clone the repository:

````bash
$ git clone https://github.com/mvivopas/RSNA-2023.git
````

2. Navigate to the repository directory and create a virtual environment

````bash
# Navigate to repository folder
$ cd RSNA-2023
# Create environment using conda
$ conda create --name rsna python=3.11
$ conda activate rsna
````

3. Install the required dependencies using pip:

````bash
$ pip install -r requirements.txt
````


## File Structure

````
- dataloaders/
    - dataloader.py
- preprocessing/
    - data_preprocessor.py
- requirements.txt
- README.md
````

- `dataloaders/`: This directory is dedicated to data loading functionality. This folder contains custom data loading code to fetch, preprocess, and organize your datasets.
- `preprocessing/`: This directory is focused on data preprocessing tasks. Data preprocessing in this case involves transforming dicom images to nifti and segmenting this niftis into diffent organs.


## Usage

1. Modify the values in the `arguments.json` to match the location where data is saved.

2. Run the preprocessor script. If the process is interrumped before finishing, simoply re-run the command and the process will resume to the last subject-sesion before interruption.

````bash
python preprocessing/data_preprocessor.py
````

## Contributing

Contributions to this repository are welcome. If you find any bugs or have suggestions for improvements, feel free to create issues or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.




[kaggle-url]: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection
