## STEP 3 ##

# flake8: noqa

# this is last part of the preprocessing pipeline
# its goal is to extract the patches from the nii volumes based on the segmentation 


import os
import sys
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
import torchio as tio

# function to extract patches from the discs in the nii folder for axial patches
def patch_extraction_volume(vol, mask, affine):
    """
    Extract a 3D patch from an MRI volume with specific real-world dimensions.
    
    Parameters:
    - vol: 3D numpy array representing the volume
    - mask: 3D segmentation mask 
    - affine: Affine matrix from the NIfTI file
    - header: Header from the NIfTI file
    
    Returns:
    - patch: 3D numpy array with specified real-world dimensions
    """
    # Convert mask to tensor for non-zero index extraction
    mask = torch.Tensor(mask)
    nonzero_indices = torch.nonzero(mask)
    
    # Calculate the centroid of the mask
    centroid = nonzero_indices.float().mean(0).numpy().astype(int)
    
    # Get voxel sizes from the affine matrix
    voxel_sizes = np.abs(np.diag(affine)[:3])
    
    # Calculate the number of voxels corresponding to 2.5 cm posterior displacement
    posterior_displacement_cm = 20
    posterior_displacement_voxels = (posterior_displacement_cm / voxel_sizes[1]).astype(int)
    
    # Compute the new centroid with posterior displacement
    # Assuming the third dimension (index 2) is the posterior-anterior axis
    displaced_centroid = centroid.copy()
    displaced_centroid[1] -= posterior_displacement_voxels
    
    # Define desired patch sizes in cm
    patch_sizes_cm = {
        'RL': 60,  # Right-Left 
        'AP': 40,  # Anterior-Posterior
        'SI': 30   # Superior-Inferior
    }
    
    # Calculate patch size in voxels
    patch_sizes_voxels = np.floor(np.array([
        patch_sizes_cm['RL'] / voxel_sizes[0],
        patch_sizes_cm['AP'] / voxel_sizes[1], 
        patch_sizes_cm['SI'] / voxel_sizes[2]
    ])).astype(int)

    # Extract patch
    D, H, W = vol.shape
    half_sizes = patch_sizes_voxels // 2
    
    patch = vol[
        max(0, displaced_centroid[0] - half_sizes[0]):min(D, displaced_centroid[0] + half_sizes[0] + patch_sizes_voxels[0] % 2),
        max(0, displaced_centroid[1] - half_sizes[1]):min(H, displaced_centroid[1] + half_sizes[1] + patch_sizes_voxels[1] % 2),
        max(0, displaced_centroid[2] - half_sizes[2]):min(W, displaced_centroid[2] + half_sizes[2] + patch_sizes_voxels[2] % 2)
    ]

    return patch

# function for sagittal patches
def patch_extraction_volume_foraminal(vol, mask, affine):
    """
    Extract two 3D patches from an MRI volume centered around mask's centroid
    
    Parameters:
    - vol: 3D numpy array representing the volume
    - mask: 3D segmentation mask 
    - affine: Affine matrix from the NIfTI file
    
    Returns:
    - patch1, patch2: Two 3D numpy array patches
    """
    i = 0

    D, H, W = vol.shape
    mask = torch.Tensor(mask)
    nonzero_indices = torch.nonzero(mask)
    
    # Calculate centroid of the mask
    centroid = nonzero_indices.float().mean(0).numpy().astype(int)

    # Get voxel sizes from the affine matrix
    voxel_sizes = np.abs(np.diag(affine)[:3])
    
    # Displacement parameters
    displacement_cm = 20  # 20 cm displacement
    posterior_displacement_voxels = (displacement_cm / voxel_sizes[1]).astype(int)
    
    patch_size_mm = {
        'd': 50,  # depth
        'h': 50,  # height
        'w': 50   # width
    }

    # Patch sizes (in voxels)
    patch_sizes_voxels = {
        'd': (patch_size_mm['d'] / voxel_sizes[0]).astype(int),
        'h': (patch_size_mm['h'] / voxel_sizes[1]).astype(int),
        'w': (patch_size_mm['w'] / voxel_sizes[2]).astype(int)
    }

    column_shift = (10/voxel_sizes[0]).astype(int)
    upper_shift = (10/voxel_sizes[1]).astype(int)
    
    # Extract patches centered on centroid with posterior displacement
    patch1 = vol[
        max(0, centroid[0] + column_shift):min(D, centroid[0] + column_shift + patch_sizes_voxels['d']//2),
        max(0, centroid[1] - posterior_displacement_voxels - patch_sizes_voxels['h']//2):min(H, centroid[1] - posterior_displacement_voxels + patch_sizes_voxels['h']//2),
        max(0, centroid[2] - patch_sizes_voxels['w']//2 + upper_shift):min(W, centroid[2] + patch_sizes_voxels['w']//2 + upper_shift)
    ]
    
    patch2 = vol[
        max(0, centroid[0] - column_shift - patch_sizes_voxels['d']//2):min(D, centroid[0] - column_shift),
        max(0, centroid[1] - posterior_displacement_voxels - patch_sizes_voxels['h']//2):min(H, centroid[1] - posterior_displacement_voxels + patch_sizes_voxels['h']//2),
        max(0, centroid[2] - patch_sizes_voxels['w']//2 + upper_shift):min(W, centroid[2] + patch_sizes_voxels['w']//2 + upper_shift)
    ]
    
    return patch1, patch2

# uses lists of sagittal images and segmentations to extract patches for each disc
def extract_and_save_sagittal_patches(sagittal_images, sagittal_segmentations, nii_folder, output_folder):
    # Match each axial image to its corresponding sagittal segmentation
    for img_name, seg_sag_name in zip(sagittal_images, sagittal_segmentations):
        if "patch" not in img_name:
            img_path = os.path.join(nii_folder, img_name)
            seg_sag_path = os.path.join(nii_folder, seg_sag_name)
            affine_ex = nib.load(img_path).affine
            
            # Load the volumetric image and sagittal segmentation
            vol = nib.load(img_path).get_fdata()
            seg_sag = nib.load(seg_sag_path).get_fdata()

            # Détection des disques dans la segmentation sagittale
            #The values to check are based on the classes in totalspineseg 
            disc_l5 = np.isin(seg_sag, [100]).astype(int)
            disc_l4 = np.isin(seg_sag, [95]).astype(int)
            disc_l3 = np.isin(seg_sag, [94]).astype(int)
            disc_l2 = np.isin(seg_sag, [93]).astype(int)
            disc_l1 = np.isin(seg_sag, [92]).astype(int)

            
            discs_dict = {
                "L1_L2": disc_l1,
                "L2_L3": disc_l2,
                "L3_L4": disc_l3,
                "L4_L5": disc_l4,
                "L5_S1": disc_l5
            }    

            # Extract and save patches for each disc
            for disc_name, disc_mask in discs_dict.items():
                if np.any(disc_mask):  # If the disc is found in the segmentation
                    # Extract the patch using the segmentation mask
                    
                    patch_img_left, patch_img_right = patch_extraction_volume_foraminal(vol, disc_mask, affine_ex)
                    
                    if patch_img_left is not None or patch_img_right is not None:  # Proceed only if patch extraction was successful

                        # Construct the filename and file path
                        patch_img_filename_left = f"{img_name[:-7]}_{disc_name}_foramen_left_patch.nii.gz"
                        patch_img_filepath_left = os.path.join(output_folder, patch_img_filename_left)

                        patch_img_filename_right = f"{img_name[:-7]}_{disc_name}_foramen_right_patch.nii.gz"
                        patch_img_filepath_right = os.path.join(output_folder, patch_img_filename_right)

                        # Use the affine from the original volume to create the patch NIfTI image
                        original_affine = nib.load(img_path).affine
                        original_header = nib.load(img_path).header.copy()
                        patch_nifti_img_left = nib.Nifti1Image(patch_img_left, affine=original_affine)
                        patch_nifti_img_right = nib.Nifti1Image(patch_img_right, affine=original_affine)


                        q_code = int(original_header['qform_code'])
                        s_code = int(original_header['sform_code'])

                        patch_nifti_img_left.header.set_qform(original_affine, code=q_code)
                        patch_nifti_img_left.header.set_sform(original_affine, code=s_code)
                        patch_nifti_img_right.header.set_qform(original_affine, code=q_code)
                        patch_nifti_img_right.header.set_sform(original_affine, code=s_code)

                        # Save the patch to the specified location
                        nib.save(patch_nifti_img_left, patch_img_filepath_left)
                        nib.save(patch_nifti_img_right, patch_img_filepath_right)


# uses lists of axial images and segmentations to extract patches for each disc
def extract_and_save_axial_patches(axial_images, axial_segmentations, nii_folder, output_folder):
    # Match each axial image to its corresponding sagittal segmentation
    for img_name, seg_sag_name in zip(axial_images, axial_segmentations):
        if "patch" not in img_name:
            img_path = os.path.join(nii_folder, img_name)
            seg_sag_path = os.path.join(nii_folder, seg_sag_name)
            
            # Load the volumetric image and sagittal segmentation
            vol = nib.load(img_path).get_fdata()
            seg_sag = nib.load(seg_sag_path).get_fdata()
            affine = nib.load(img_path).affine

            # Détection des disques dans la segmentation sagittale
            #The values to check are based on the classes in totalspineseg 
            disc_l5 = np.isin(seg_sag, [100]).astype(int)
            disc_l4 = np.isin(seg_sag, [95]).astype(int)
            disc_l3 = np.isin(seg_sag, [94]).astype(int)
            disc_l2 = np.isin(seg_sag, [93]).astype(int)
            disc_l1 = np.isin(seg_sag, [92]).astype(int)

            
            discs_dict = {
                "L1_L2": disc_l1,
                "L2_L3": disc_l2,
                "L3_L4": disc_l3,
                "L4_L5": disc_l4,
                "L5_S1": disc_l5
            }

            # Extract and save patches for each disc
            for disc_name, disc_mask in discs_dict.items():
                if np.any(disc_mask):  # If the disc is found in the segmentation
                    # Extract the patch using the segmentation mask
                    
                    patch_img = patch_extraction_volume(vol, disc_mask, affine)
                    
                    if patch_img is not None:  # Proceed only if patch extraction was successful

                        # Construct the filename and file path
                        patch_img_filename = f"{img_name[:-7]}_{disc_name}_patch.nii.gz"
                        patch_img_filepath = os.path.join(output_folder, patch_img_filename)
                        
                        # Use the affine from the original volume to create the patch NIfTI image
                        original_affine = nib.load(img_path).affine
                        patch_nifti_img = nib.Nifti1Image(patch_img, affine=original_affine)

                        original_header = nib.load(img_path).header.copy()

                        q_code = int(original_header['qform_code'])
                        s_code = int(original_header['sform_code'])

                        patch_nifti_img.header.set_qform(original_affine, code=q_code)
                        patch_nifti_img.header.set_sform(original_affine, code=s_code)

                        # Save the patch to the specified location
                        nib.save(patch_nifti_img, patch_img_filepath)

# extract patches from the discs in the nii folder, for axial and sagittal patches
def extract_patches_from_discs(nii_folder, output_folder):
    """
    Traverses a folder containing MRIs and associated sagittal segmentations.
    For each axial image and associated sagittal segmentation, extracts patches for discs with labels 206 to 202.
    Saves each patch in the corresponding folder structure within output_folder.

    nii_folder : path to the folder containing MRIs and segmentations
    output_folder : path to the folder where patches will be saved
    """
    axial_images = []
    axial_segmentations = []
    sagittal_T2_segmentations = []
    sagittal_T1_segmentations = []
    saggital_T1_images = []
    sagittal_T2_images = []

    
    # Traverse files in the nii_folder
    for filename in os.listdir(nii_folder):
        if 'acq-ax' in filename and filename.endswith('.nii.gz') and not filename .endswith('_seg.nii.gz'):          
            axial_images.append(filename)  # Axial images
        elif 'acq-ax' in filename and 'T2w' in filename and 'total_seg.nii.gz' in filename:
            axial_segmentations.append(filename)  # Sagittal segmentations
        elif 'acq-sag' in filename and 'T2w' in filename and 'total_seg.nii.gz' in filename:
            sagittal_T2_segmentations.append(filename)
        elif 'acq-sag' in filename and 'T1w' in filename and 'total_seg.nii.gz' in filename:
            sagittal_T1_segmentations.append(filename)
        elif 'acq-sag' in filename and 'T2w' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
            sagittal_T2_images.append(filename)
        elif 'acq-sag' in filename and 'T1w' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
            saggital_T1_images.append(filename)

    # Sort lists to ensure corresponding order
    axial_segmentations.sort()
    axial_images.sort()
    sagittal_T2_segmentations.sort()
    sagittal_T2_images.sort()
    sagittal_T1_segmentations.sort()
    saggital_T1_images.sort()
    sagittal_T2_segmentations.sort()

    extract_and_save_sagittal_patches(sagittal_T2_images, sagittal_T2_segmentations, nii_folder, output_folder)
    extract_and_save_sagittal_patches(saggital_T1_images, sagittal_T1_segmentations, nii_folder, output_folder)
    extract_and_save_axial_patches(axial_images, axial_segmentations, nii_folder, output_folder)

# function to select the patches with the best resolution for each disc
def select_best_patches(folder_path):
    discs = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
    disc_patches = {disc: [] for disc in discs}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz') and '_seg' not in filename and "acq-ax" in filename:
            for disc in discs:
                if f"{disc}_patch" in filename:
                    file_path = os.path.join(folder_path, filename)
                    img = nib.load(file_path)
                    resolution = img.header.get_zooms()
                    voxel_volume = resolution[0] * resolution[1] * resolution[2]

                    disc_patches[disc].append((file_path, voxel_volume))
                 
    for disc, patches in disc_patches.items():
        if len(patches) > 1:
            # Sort patches by increasing voxel volume (resolution)
            patches.sort(key=lambda x: x[1])
            
            # Keep the patch with the best resolution
            best_patch = patches[0]
            
            # Remove other patches and their segmentations
            for patch in patches[1:]:
                os.remove(patch[0])

# function to process all subjects in a directory
def process_all_subjects_in_directory(root_dir, output_root_dir):
    """
    Traverses all subdirectories in the root directory corresponding to subjects,
    and applies the patch extraction function to each subdirectory.
    
    root_dir : root directory containing subject subdirectories
    output_root_dir : root directory where output patches are stored
    """
    for subject_folder in os.listdir(root_dir):
        
        subject_path = os.path.join(root_dir, subject_folder, "anat")
        output_subject_path = os.path.join(output_root_dir, subject_folder, "anat")
        
        # Check if it is a subdirectory
        if os.path.isdir(subject_path):
            os.makedirs(output_subject_path, exist_ok=True)
            # Apply the patch extraction function to each subject
            extract_patches_from_discs(subject_path, output_subject_path)

            # Apply the function to select the best patches if there are multiple ones for the same disc
            select_best_patches(output_subject_path)
        

folder = "C:/Users/abels/OneDrive/Documents/NeuroPoly/rsna-challenge/test_bids"

def main():
    # Ensure a directory argument is passed
    if len(sys.argv) != 2:
        print("Usage: python extraction.py [data_directory]")
        sys.exit(1)
    
    # Get the root directory from the command-line argument
    root_dir = sys.argv[1]
    output_dir = sys.argv[1]
    
    
    # in case of registration of seg needed
    # process_directory_other(root_dir)


    # Run the processing function for all subjects in the specified directory
    process_all_subjects_in_directory(root_dir, output_dir)

