import os
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import subprocess

# Define the base directory where your BIDS data is stored
#base_dir = '/media/xkrejc78/Transcend/NeuroPoly_internship/data/rsna-2024-lumbar-challenge/bids-rsna-lscd'
base_dir = '/home/ge.polymtl.ca/p120942/data/bids-rsna-lscd'


# Function to resample a NIfTI image to a new voxel size
def resample_image_to_voxel_size(nii_file, new_voxel_size):
    # Load the NIfTI image
    img = nib.load(nii_file)
    data = img.get_fdata()
    header = img.header

    # Get the current voxel size
    current_voxel_size = header.get_zooms()[:3]

    # Calculate the resampling factors
    factors = [cv / nv for cv, nv in zip(current_voxel_size, new_voxel_size)]

    # Resample the image using zoom
    resampled_data = zoom(data, factors, order=3)  # order=3 for cubic interpolation

    # Create a new affine matrix to reflect the new voxel size
    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[i, i] = new_voxel_size[i]

    # Create a new NIfTI image
    resampled_img = nib.Nifti1Image(resampled_data, new_affine)

    return resampled_img


# Function to iterate through subjects and their anat folders, resample images to new voxel size, and save them
def iterate_and_resample_bids_anat_files(base_dir):
    # Find all subject directories (sub-*)
    subjects = glob.glob(os.path.join(base_dir, 'sub-*'))
    T1w_voxel_sizes = []
    T2w_voxel_sizes_sag = []
    T2w_voxel_sizes_ax = []

    for subject in subjects:
        # Define the anat directory for the current subject
        anat_dir = os.path.join(subject, 'anat')

        # Check if the anat directory exists
        if os.path.exists(anat_dir):
            # Find all .nii.gz files in the anat directory
            nii_files = glob.glob(os.path.join(anat_dir, '*.nii.gz'))

            for nii_file in nii_files:
                # Print the current file being processed
                print(f"Processing {nii_file}")

                # Load the NIfTI image
                img = nib.load(nii_file)

                # get the current voxel size
                current_voxel_size = img.header.get_zooms()[:3]

                # if the nii_file ends with T1w.nii.gz, save the current voxel size for list:
                if nii_file.endswith('T1w.nii.gz'):
                    T1w_voxel_sizes.append(current_voxel_size)

                # if the nii_file ends with T2w.nii.gz and in the name is "sag", save the current voxel size for list:
                elif nii_file.endswith('T2w.nii.gz') and 'sag' in nii_file:
                    T2w_voxel_sizes_sag.append(current_voxel_size)

                # if the nii_file ends with T2w.nii.gz and in the name is "ax", save the current voxel size for list:
                elif nii_file.endswith('T2w.nii.gz') and 'ax' in nii_file:
                    T2w_voxel_sizes_ax.append(current_voxel_size)

    
    # get median of the voxel sizes for each type of image
    T1w_voxel_sizes_median = np.median(T1w_voxel_sizes, axis=0)
    T2w_voxel_sizes_sag_median = np.median(T2w_voxel_sizes_sag, axis=0)
    T2w_voxel_sizes_ax_median = np.median(T2w_voxel_sizes_ax, axis=0)

    for subject in subjects:
        # Define the anat directory for the current subject
        anat_dir = os.path.join(subject, 'anat')
        non_resampled = os.path.join(anat_dir, 'non-resampled')
        resampled = os.path.join(anat_dir, 'resampled')

        # Check if the anat directory exists
        if os.path.exists(anat_dir):
            # Find all .nii.gz files in the anat directory
            nii_files = glob.glob(os.path.join(anat_dir, '*.nii.gz'))

            for nii_file in nii_files:
                # Print the current file being processed
                print(f"Processing {nii_file}")

                # Load the NIfTI image
                img = nib.load(nii_file)

                # if the nii_file ends with T1w.nii.gz, save the current voxel size for list:
                if nii_file.endswith('T1w.nii.gz'):
                    resampled_img = resample_image_to_voxel_size(nii_file, T1w_voxel_sizes_median)
                    output_path = nii_file.replace('.nii.gz', '_resampled.nii.gz')
                    nib.save(resampled_img, output_path)

                # if the nii_file ends with T2w.nii.gz and in the name is "sag", save the current voxel size for list:
                elif nii_file.endswith('T2w.nii.gz') and 'sag' in nii_file:
                    resampled_img = resample_image_to_voxel_size(nii_file, T2w_voxel_sizes_sag_median)
                    output_path = nii_file.replace('.nii.gz', '_resampled.nii.gz')
                    nib.save(resampled_img, output_path)

                # if the nii_file ends with T2w.nii.gz and in the name is "ax", save the current voxel size for list:
                elif nii_file.endswith('T2w.nii.gz') and 'ax' in nii_file:
                    resampled_img = resample_image_to_voxel_size(nii_file, T2w_voxel_sizes_ax_median)
                    output_path = nii_file.replace('.nii.gz', '_resampled.nii.gz')
                    nib.save(resampled_img, output_path)


                output_path_seg_resampled = nii_file.replace('.nii.gz', '_resampled_seg.nii.gz')
                output_path_seg = nii_file.replace('.nii.gz', '_seg.nii.gz')

                # Construct the TotalSegmentator command - for non-resampled image
                command = [
                    'TotalSegmentator',
                    '-i', output_path,
                    '-o', non_resampled,
                    '--task', 'total_mr'
                ]

                # Run the TotalSegmentator command
                print(f"Running TotalSegmentator for file: {resampled_img}")
                subprocess.run(command, check=True)

                # Construct the TotalSegmentator command - for resampled image
                command = [
                    'TotalSegmentator',
                    '-i', nii_file,
                    '-o', resampled,
                    '--task', 'total_mr'
                ]
                #
                # Run the TotalSegmentator command
                print(f"Running TotalSegmentator for file: {nii_file}")
                subprocess.run(command, check=True)
                #
                print(f"Segmentation completed for file: {nii_file}, saved in {output_path_seg}")


# Call the function to start the iteration and resampling
iterate_and_resample_bids_anat_files(base_dir)
