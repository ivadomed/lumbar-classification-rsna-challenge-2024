"""
Take all the dataset and convert it to nifti format using dcm2niix.
The nifti files will be saved in the same folder structure as the original dataset.
"""
import os
import numpy as np
import subprocess
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Convert dataset from dcm to nifti format')
parser.add_argument('--input_folder', type=str, help='Path to the dataset in dcm format')
args = parser.parse_args()

input_folder = args.input_folder

# make sure the input folder end with "train_images"
if input_folder.endswith("train_images"):
    dataset_type='train'
elif input_folder.endswith("test_images"):
    dataset_type='test'
else:
    print("The input folder should end with 'train_images' or 'test_images'")
    exit()

processed_folders = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".dcm"):
            if root not in processed_folders:
                processed_folders.append(root)

                root_nifti = root.replace(f"{dataset_type}_images", f"{dataset_type}_nifti")
                if not os.path.exists(root_nifti):
                    os.makedirs(root_nifti)
                subprocess.run(f'dcm2niix -o {root_nifti} -z y {root}', shell=True)
