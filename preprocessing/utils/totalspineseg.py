"""
This script is used to perform the segmentation of the scans using TotalSpineSeg.
It has to be used after niftification.py and niftification.py 

Input: 
    --data: Path to the root directory of the dataset.
Output:
    None

Author: Thomas Dagonneau and Abel Salmona 
"""

import os
import shutil
import sys
import csv
import argparse



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the root directory of the dataset.")
    args = parser.parse_args()
    return args

def get_batches(source_dir, batch_size=50):
    """Get a list of batches of subjects filtered by the filter_func."""
    all_subjects = [sub for sub in os.listdir(source_dir)]
    return [all_subjects[i:i + batch_size] for i in range(0, len(all_subjects), batch_size)]


def run_totalspineseg(source_dir):
    """Applies TotalSpineSeg to every scan in the source_dir and saves the segmentations."""
    # Define temporary directories
    tss_temp_dir = "data"
    output_temp = "temp_output_data"
    failed_subjects = []

    # Get all batches of subjects
    batches = get_batches(source_dir, batch_size=50)

    # Process each batch
    for i, batch in enumerate(batches):
        os.makedirs(tss_temp_dir, exist_ok=True)
        os.makedirs(output_temp, exist_ok=True)

        for subdir in batch:
            try:
                anat_path = os.path.join(source_dir, subdir, 'anat')
                if os.path.exists(anat_path):
                    for file in os.listdir(anat_path):
                        file_path = os.path.join(anat_path, file)
                        if os.path.isfile(file_path) and 'sag' in file_path and 'total_seg' not in file_path and file_path.endswith('.nii.gz'):
                            print(file_path)
                            shutil.copy(file_path, tss_temp_dir)
                            print('File copied successfully.')
            except Exception as e:
                print(f"Failed processing subject {subdir}: {e}")
                failed_subjects.append(subdir)

        # Run TotalSpineSeg segmentation
        os.system(f'totalspineseg {tss_temp_dir} {output_temp} --step1')

        # Move segmentations back into original data structure
        segmentations_into_anat(output_temp, source_dir)

        # Clean up temporary directories
        shutil.rmtree(tss_temp_dir)
        shutil.rmtree(output_temp)


def segmentations_into_anat(output_folder, nii_folder):
    """Send the segmentations into the folder with the nii volumes."""
    seg_folder = os.path.join(output_folder, "step1_output")
    segmentations = os.listdir(seg_folder)

    for segmentation in segmentations:
        id_patient = segmentation.split('_')[0]
        patient_folder = os.path.join(nii_folder, id_patient, 'anat')

        if os.path.exists(patient_folder):
            source_path = os.path.join(seg_folder, segmentation)
            modified_segmentation = segmentation.replace('.nii.gz', '_total_seg.nii.gz')
            destination_path = os.path.join(patient_folder, modified_segmentation)
            shutil.copy(source_path, destination_path)


def main():
    
    args = parse_arguments()
    data_directory = args.data

    run_totalspineseg(data_directory)


if __name__ == "__main__":
    main()
