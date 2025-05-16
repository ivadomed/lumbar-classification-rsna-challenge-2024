import os
import sys
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
import torchio as tio
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops
import pandas as pd
import ast
from itertools import product

def is_point_in_patch(x, y, z, patch_slices):
    d_slice = patch_slices[0]
    h_slice = patch_slices[1]
    w_slice = patch_slices[2]
    
    return (
        d_slice[0] <= x <= d_slice[1] and
        h_slice[0] <= y <= h_slice[1] and
        w_slice[0] <= z <= w_slice[1]
    )


# function to calculate the com of a disk and shift it along the disk axis (for the foramina)
# this fdunction is not really time consuming (I was worried for regionprops but it's a matter of ms)
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops
import nibabel as nib

def get_shifted_point_along_disk(disk_mask, affine):
    """
    Compute a shifted point along the disk's anatomical axis, accounting for scan orientation.

    Args:
        disk_mask: 3D binary mask of the disk
        affine: affine matrix of the image

    Returns:
        shifted_point: numpy array of (x, y, z) voxel coordinates of the shifted point
        disk_radius: approximate radius along disk axis (in voxels)
        direction_vector: unit vector showing shift direction (in voxel space)
    """
    # Compute the center of mass in voxel space
    centroid = center_of_mass(disk_mask)
    # Determine anatomical orientation codes
    orientation = nib.aff2axcodes(affine)  # e.g., ('A', 'I', 'L') or ('L', 'P', 'I')
    
    # Find which axis is sagittal (LR), coronal (AP), axial (IS)
    sagittal_axis = orientation.index('L') if 'L' in orientation else orientation.index('R')
    coronal_axis  = orientation.index('P') if 'P' in orientation else orientation.index('A')
    axial_axis    = orientation.index('I') if 'I' in orientation else orientation.index('S')
    
    # Extract the sagittal slice at centroid
    sagittal_slice_idx = int(centroid[sagittal_axis])
    slicer = [slice(None)] * 3
    slicer[sagittal_axis] = sagittal_slice_idx
    sagittal_slice = disk_mask[tuple(slicer)]

    # Collapse to 2D (coronal and axial)
    axes_2d = [i for i in range(3) if i != sagittal_axis]
    sagittal_2d = np.transpose(sagittal_slice, axes_2d)

    # Regionprops for orientation
    props = regionprops(sagittal_2d.astype(int))
    if not props:
        return centroid, 0, np.zeros(3)  # fallback if empty

    orientation_angle = props[0].orientation  # radians

    # Construct direction vector in voxel space
    direction_vector = np.zeros(3)
    direction_vector[axes_2d[0]] = -np.cos(orientation_angle)  # coronal axis
    direction_vector[axes_2d[1]] = -np.sin(orientation_angle)  # axial axis
    # Flip direction vector to match LPI reference
    if orientation[coronal_axis] == 'P':  # anterior = negative in LPI
        direction_vector[coronal_axis] *= -1
    if orientation[axial_axis] == 'S':  # superior = negative in LPI
        direction_vector[axial_axis] *= -1

    # Normalize
    direction_vector /= np.linalg.norm(direction_vector)

    # Estimate radius along direction vector
    mask_points = np.array(np.where(disk_mask)).T
    centered_points = mask_points - centroid
    projections = np.abs(centered_points @ direction_vector)
    disk_radius = np.max(projections)

    # Compute shifted point
    shifted_point = centroid + direction_vector * disk_radius

    return shifted_point


# function for sagittal patches
def patch_extraction_foraminal(vol, mask, affine, pixdim):
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
    # mask = torch.Tensor(mask)
    # nonzero_indices = torch.nonzero(mask)
    
    # Calculate centroid of the mask and shift it along the disk axis
    centroid = get_shifted_point_along_disk(mask, affine).astype(int)


    # Get voxel sizes from the affine matrix
    voxel_sizes = pixdim
    
        # Define patch size in cm
    patch_size_mm = {
        'd': 50,  # depth (along IS)
        'h': 50,  # height (along AP)
        'w': 50   # width (along LR)
    }

    # Get voxel sizes from the affine matrix
    
    # Patch sizes in voxels
    patch_sizes_voxels = {
        0: int(patch_size_mm['d'] / voxel_sizes[0]),
        1: int(patch_size_mm['h'] / voxel_sizes[1]),
        2: int(patch_size_mm['w'] / voxel_sizes[2])
    }

    

    # Determine which axis corresponds to LR and its direction
    orientation = nib.aff2axcodes(affine)  # e.g., ('A', 'I', 'L') or ('L', 'P', 'I')
    # Find which axis is sagittal (LR), coronal (AP), axial (IS)
    lr_axis = orientation.index('L') if 'L' in orientation else orientation.index('R')

    lr_sign = np.sign(affine[lr_axis, 0])       # negative if increasing index goes L→R
    #print(lr_axis, lr_sign)


    # Shift centroid in both LR directions
    centroid = centroid.astype(int)
    
    patch1 = [[],[],[]]
    patch2 = [[],[],[]]
    
    for i in range(3):
        if i == lr_axis:
            patch1[i] = [max(0, centroid[i] + 1), min(D, centroid[i] + patch_sizes_voxels[i] // 2)]
            patch2[i] = [max(0, centroid[i] - 1 - patch_sizes_voxels[i] // 2), min(D, centroid[i] -1)]
        else:
            patch1[i] = [max(0, centroid[i] - patch_sizes_voxels[i] // 2), min(D, centroid[i] + patch_sizes_voxels[i] // 2)]
            patch2[i] = [max(0, centroid[i] - patch_sizes_voxels[i] // 2), min(D, centroid[i] + patch_sizes_voxels[i] // 2)]

    
    return patch1, patch2
    
    

# uses lists of sagittal images and segmentations to extract patches for each disc
def extract_and_save_sagittal_patches(sagittal_images, sagittal_segmentations, nii_folder, output_folder):
    results = []
    df = pd.read_csv("processed_annotations.csv")
    TP=0 
    total=0 
    # Match each axial image to its corresponding sagittal segmentation
    for img_name, seg_sag_name in zip(sagittal_images, sagittal_segmentations):
        if "patch" not in img_name:
            study_id = (img_name.split("-")[1].split("_")[0])
            
            img_path = os.path.join(nii_folder, img_name)
            seg_sag_path = os.path.join(nii_folder, seg_sag_name)
            affine_ex = nib.load(img_path).affine
            
            # Load the volumetric image and sagittal segmentation
            vol = nib.load(img_path).get_fdata()
            D, H, W = vol.shape
            seg_sag = nib.load(seg_sag_path).get_fdata()
            pixdim = nib.load(img_path).header['pixdim'][1:4]  # Get pixel dimensions

            # Détection des disques dans la segmentation sagittale
            #The values to check are based on the classes in totalspineseg 
            disc_l5 = np.isin(seg_sag, [100]).astype(int)
            disc_l4 = np.isin(seg_sag, [95]).astype(int)
            disc_l3 = np.isin(seg_sag, [94]).astype(int)
            disc_l2 = np.isin(seg_sag, [93]).astype(int)
            disc_l1 = np.isin(seg_sag, [92]).astype(int)

            
            discs_dict = {
                "L1/L2": disc_l1,
                "L2/L3": disc_l2,
                "L3/L4": disc_l3,
                "L4/L5": disc_l4,
                "L5/S1": disc_l5
            }    

            # Extract and save patches for each disc
            for disc_name, disc_mask in discs_dict.items():
                if np.any(disc_mask):  # If the disc is found in the segmentation
                    # Extract the patch using the segmentation mask
                    
                    patch_img_right, patch_img_left = patch_extraction_foraminal(vol, disc_mask, affine_ex, pixdim)
                    
                    if patch_img_left is not None or patch_img_right is not None:  # Proceed only if patch extraction was successful

                        try:
                            point_left = df.loc[
                                (df['study_id'] == int(study_id)) & 
                                (df['level'] == disc_name) & 
                                (df['condition'] == 'Left Neural Foraminal Narrowing')
                            ].iloc[0]

                            point_right = df.loc[
                                (df['study_id'] == int(study_id)) & 
                                (df['level'] == disc_name) & 
                                (df['condition'] == 'Right Neural Foraminal Narrowing')
                            ].iloc[0]

                            point_data = [
                                ('Left Neural Foraminal Narrowing', point_left, patch_img_left),
                                ('Right Neural Foraminal Narrowing', point_right, patch_img_right)
                            ]
                        
                        
                            #for condition, point, patch_slices in point_data:

                            condition_left, point_left, patch_slices_left = point_data[0]
                            condition_right, point_right, patch_slices_right = point_data[1]

                            L_left = round(float(point_left['L']), 2) -1
                            P_left = round(float(point_left['P']), 2) 
                            I_left = round(float(point_left['I']), 2)
                            
                            new_P_left = D - P_left 
                            new_I_left = H - I_left
                            new_L_left = W - L_left - 1 

                            L_right = round(float(point_right['L']), 2) -1
                            P_right = round(float(point_right['P']), 2)
                            I_right = round(float(point_right['I']), 2)

                            new_P_right = D - P_right 
                            new_I_right = H - I_right 
                            new_L_right = W - L_right -1 
                            

                            #inside_left = is_point_in_patch(new_P_left , new_I_left , new_L_left ,  patch_slices_left ) or  is_point_in_patch( new_P_left, I_left, new_L_left, patch_slices_left) or is_point_in_patch(P_left, new_I_left, new_L_left, patch_slices_left) or is_point_in_patch( P_left , I_left , L_left , patch_slices_left ) or is_point_in_patch(new_P_left , new_I_left , L_left ,  patch_slices_left ) or   is_point_in_patch( new_P_left, I_left, L_left, patch_slices_left) or is_point_in_patch(P_left, new_I_left, L_left, patch_slices_left) or is_point_in_patch( P_left , I_left , new_L_left , patch_slices_left )
                            #inside_right = is_point_in_patch(new_P_right , new_I_right , new_L_right ,  patch_slices_right ) or  is_point_in_patch( new_P_right, I_right, new_L_right, patch_slices_right) or is_point_in_patch(P_right, new_I_right, new_L_right, patch_slices_right) or is_point_in_patch( P_right , I_right , L_right , patch_slices_right ) or is_point_in_patch(new_P_right , new_I_right , L_right ,  patch_slices_right ) or   is_point_in_patch( new_P_right, I_right, L_right, patch_slices_right) or is_point_in_patch(P_right, new_I_right, L_right, patch_slices_right) or is_point_in_patch( P_right , I_right , new_L_right , patch_slices_left )

                            

                            

                            # Define coordinate options
                            P_left_opts  = [P_left,  new_P_left]
                            I_left_opts  = [I_left,  new_I_left]
                            L_left_opts  = [L_left,  new_L_left]

                            P_right_opts = [P_right, new_P_right]
                            I_right_opts = [I_right, new_I_right]
                            L_right_opts = [L_right, new_L_right]

                            # === First test: original patch assignment ===

                            valid_left_combos = set()
                            valid_right_combos = set()

                            for p, i, l in product([0, 1], repeat=3):
                                if is_point_in_patch(P_left_opts[p], I_left_opts[i], L_left_opts[l], patch_slices_left):
                                    valid_left_combos.add((p, i, l))
                                if is_point_in_patch(P_right_opts[p], I_right_opts[i], L_right_opts[l], patch_slices_right):
                                    valid_right_combos.add((p, i, l))

                            common_valid_combos = valid_left_combos & valid_right_combos

                            inside_left = inside_right = False

                            if common_valid_combos:
                                inside_left = inside_right = True
                            elif valid_left_combos:
                                inside_left = True
                            elif valid_right_combos:
                                inside_right = True

                            # === Second test: switched patch assignment ===

                            valid_left_combos_swapped = set()
                            valid_right_combos_swapped = set()

                            for p, i, l in product([0, 1], repeat=3):
                                if is_point_in_patch(P_left_opts[p], I_left_opts[i], L_left_opts[l], patch_slices_right):  # swapped!
                                    valid_left_combos_swapped.add((p, i, l))
                                if is_point_in_patch(P_right_opts[p], I_right_opts[i], L_right_opts[l], patch_slices_left):  # swapped!
                                    valid_right_combos_swapped.add((p, i, l))

                            common_swapped_combos = valid_left_combos_swapped & valid_right_combos_swapped

                            if common_swapped_combos:
                                inside_left = inside_right = True

                            
                            results.append({
                                "study_id": study_id,
                                "level": disc_name,
                                "condition": condition_left,
                                "point_in_patch": inside_left,
                                "point_L": L_left,
                                "point_P": P_left,
                                "point_I": I_left,
                                "new_point_L": new_L_left,
                                "new_point_P": new_P_left,
                                "new_point_I": new_I_left,
                                "patch_L_min": patch_slices_left[0][0],
                                "patch_L_max": patch_slices_left[0][1],
                                "patch_P_min": patch_slices_left[1][0],
                                "patch_P_max": patch_slices_left[1][1],
                                "patch_I_min": patch_slices_left[2][0],
                                "patch_I_max": patch_slices_left[2][1]
                            })

                            results.append({
                                "study_id": study_id,
                                "level": disc_name,
                                "condition": condition_right,
                                "point_in_patch": inside_right,
                                "point_L": L_right,
                                "point_P": P_right,
                                "point_I": I_right,
                                "new_point_L": new_L_right,
                                "new_point_P": new_P_right,
                                "new_point_I": new_I_right,
                                "patch_L_min": patch_slices_right[0][0],
                                "patch_L_max": patch_slices_right[0][1],
                                "patch_P_min": patch_slices_right[1][0],
                                "patch_P_max": patch_slices_right[1][1],
                                "patch_I_min": patch_slices_right[2][0],
                                "patch_I_max": patch_slices_right[2][1]
                            })

                            # Count TP/total for each condition
                            print(inside_left, inside_right)
                            TP += inside_left 
                            TP += inside_right
                            total += 2
                        except IndexError:
                            results.append({
                                "study_id": study_id,
                                "level": disc_name,
                                "condition": "Left Neural Foraminal Narrowing",
                                "point_in_patch": False,
                                "point_L": -1,
                                "point_P": -1,
                                "point_I": -1,
                                "patch_L_min": patch_img_left[0][0],
                                "patch_L_max": patch_img_left[0][1],
                                "patch_P_min": patch_img_left[1][0],
                                "patch_P_max": patch_img_left[1][1],
                                "patch_I_min": patch_img_left[2][0],
                                "patch_I_max": patch_img_left[2][1]
                            })
                            results.append({
                                "study_id": study_id,
                                "level": disc_name,
                                "condition": "Right Neural Foraminal Narrowing",
                                "point_in_patch": False,
                                "point_L": -1,
                                "point_P": -1,
                                "point_I": -1,
                                "patch_L_min": patch_img_right[0][0],
                                "patch_L_max": patch_img_right[0][1],
                                "patch_P_min": patch_img_right[1][0],
                                "patch_P_max": patch_img_right[1][1],
                                "patch_I_min": patch_img_right[2][0],
                                "patch_I_max": patch_img_right[2][1]
                            })

    df_results = pd.DataFrame(results)
    return TP, total, df_results
                        
                        

                        

# uses lists of axial images and segmentations to extract patches for each disc
def extract_and_save_axial_patches(axial_images, axial_segmentations, nii_folder, output_folder):
    df = pd.read_csv("processed_annotations.csv")
    TP = 0
    total = 0
    results = []
    
    for img_name, seg_sag_name in zip(axial_images, axial_segmentations):
        if "patch" not in img_name:
            study_id = (img_name.split("-")[1].split("_")[0])
            img_path = os.path.join(nii_folder, img_name)
            seg_sag_path = os.path.join(nii_folder, seg_sag_name)

            # Load volumes and affine
            vol = nib.load(img_path).get_fdata()
            seg_sag = nib.load(seg_sag_path).get_fdata()
            affine = nib.load(img_path).affine

            # Segment disc levels
            disc_l5 = np.isin(seg_sag, [100]).astype(int)
            disc_l4 = np.isin(seg_sag, [95]).astype(int)
            disc_l3 = np.isin(seg_sag, [94]).astype(int)
            disc_l2 = np.isin(seg_sag, [93]).astype(int)
            disc_l1 = np.isin(seg_sag, [92]).astype(int)

            discs_dict = {
                "L1/L2": disc_l1,
                "L2/L3": disc_l2,
                "L3/L4": disc_l3,
                "L4/L5": disc_l4,
                "L5/S1": disc_l5
            }
          

            for disc_name, disc_mask in discs_dict.items():
                if np.any(disc_mask):
                  
                    patch_img = patch_extraction_volume(vol, disc_mask, affine)

                    if patch_img is not None:
                        try:
                            point = df.loc[
                                (df['study_id'] == int(study_id)) &
                                (df['level'] == disc_name) &
                                (df['condition'] == 'Spinal Canal Stenosis')
                            ].iloc[0]

                            L = float(point['L'])
                            P = float(point['P'])
                            I = float(point['I'])

                            inside = is_point_in_patch(L, P, I, patch_img)

                            results.append({
                                "study_id": study_id,
                                "level": disc_name,
                                "condition": "Spinal Canal Stenosis",
                                "point_in_patch": inside,
                                "point_L": L,
                                "point_P": P,
                                "point_I": I,
                                "patch_L_min": patch_img[0][0],
                                "patch_L_max": patch_img[0][1],
                                "patch_P_min": patch_img[1][0],
                                "patch_P_max": patch_img[1][1],
                                "patch_I_min": patch_img[2][0],
                                "patch_I_max": patch_img[2][1]
                            })

                            TP += inside
                            total += 1
                        except IndexError:
                            results.append({
                                "study_id": study_id,
                                "level": disc_name,
                                "condition": "Spinal Canal Stenosis",
                                "point_in_patch": False,
                                "point_L": -1,
                                "point_P": -1,
                                "point_I": -1,
                                "patch_L_min": patch_img[0][0],
                                "patch_L_max": patch_img[0][1],
                                "patch_P_min": patch_img[1][0],
                                "patch_P_max": patch_img[1][1],
                                "patch_I_min": patch_img[2][0],
                                "patch_I_max": patch_img[2][1]
                            })

    df_results = pd.DataFrame(results)
    return TP, total, df_results

                        
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
    sagittal_T1_segmentations = []
    sagittal_T1_images = []
    

    
    # Traverse files in the nii_folder
    for filename in os.listdir(nii_folder):
        if 'acq-ax' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
            axial_images.append(filename)  # Axial images
        elif 'acq-ax' in filename and 'T2w' in filename and 'total_seg.nii.gz' in filename:
            axial_segmentations.append(filename)  # Sagittal segmentations
        elif 'acq-sag' in filename and 'T1w' in filename and 'total_seg.nii.gz' in filename:
            sagittal_T1_segmentations.append(filename)
        elif 'acq-sag' in filename and 'T1' in filename and filename.endswith('.nii.gz') and not filename.endswith('_seg.nii.gz'):          
            sagittal_T1_images.append(filename)

    # Sort lists to ensure corresponding order
    axial_segmentations.sort()
    axial_images.sort()

    sagittal_T1_segmentations.sort()
    sagittal_T1_images.sort()

    #extract_and_save_sagittal_patches(sagittal_T2_images, sagittal_T2_segmentations, nii_folder, output_folder)
    TP_nfn,total_nfn,df_nfn = extract_and_save_sagittal_patches(sagittal_T1_images, sagittal_T1_segmentations, nii_folder, output_folder)
    #TP_scs,total_scs,df_scs = extract_and_save_axial_patches(axial_images, axial_segmentations, nii_folder, output_folder)
    #df = pd.concat([df_nfn, df_scs], ignore_index=True)
    df = df_nfn 
    TP_scs = 0 
    total_scs = 1 
    return TP_nfn,total_nfn,TP_scs,total_scs, df 



def patch_extraction_volume(vol, mask, affine):
    """
    Extract a 3D patch from an MRI volume with specific real-world dimensions,
    accounting for orientation and directionality.
    
    Parameters:
    - vol: 3D numpy array (volume)
    - mask: 3D numpy array (segmentation mask)
    - affine: affine matrix from the NIfTI file
    
    Returns:
    - patch: list of [start, end] for each axis (voxel indices)
    """
    # Determine axis orientation
    axcodes = nib.aff2axcodes(affine)  # e.g. ('L', 'P', 'I')
    axmap = {dir: i for i, dir in enumerate(axcodes)}

    # Map anatomical directions to axes
    axis_rl = axmap.get('L', axmap.get('R'))
    axis_ap = axmap.get('P', axmap.get('A'))
    axis_si = axmap.get('I', axmap.get('S'))

    # Get voxel sizes
    voxel_sizes = np.abs(np.diag(affine)[:3])

    # Convert mask to tensor and compute centroid
    mask_tensor = torch.tensor(mask)
    nonzero_indices = torch.nonzero(mask_tensor)
    centroid = nonzero_indices.float().mean(0).numpy().astype(int)

    # Posterior displacement in voxel units
    posterior_displacement_cm = 20
    displacement_vox = int(posterior_displacement_cm / voxel_sizes[axis_ap])

    # Adjust displacement direction depending on axis code
    displaced_centroid = centroid.copy()
    ap_code = axcodes[axis_ap]  # 'P' or 'A'

    if ap_code == 'P':
        displaced_centroid[axis_ap] += displacement_vox
    elif ap_code == 'A':
        displaced_centroid[axis_ap] -= displacement_vox
    else:
        raise ValueError(f"Unexpected axis code for AP axis: {ap_code}")

    # Patch size in cm
    patch_sizes_cm = {'RL': 60, 'AP': 40, 'SI': 30}
    patch_sizes_voxels = np.floor([
        patch_sizes_cm['RL'] / voxel_sizes[axis_rl],
        patch_sizes_cm['AP'] / voxel_sizes[axis_ap],
        patch_sizes_cm['SI'] / voxel_sizes[axis_si]
    ]).astype(int)

    # Map patch sizes to each voxel axis
    axis_sizes = {axis_rl: patch_sizes_voxels[0],
                  axis_ap: patch_sizes_voxels[1],
                  axis_si: patch_sizes_voxels[2]}

    # Compute patch bounds
    shape = vol.shape
    patch = []
    for i in range(3):
        half = axis_sizes[i] // 2
        start = max(0, displaced_centroid[i] - half)
        end = min(shape[i], displaced_centroid[i] + half + axis_sizes[i] % 2)
        patch.append([start, end])

    print(patch)
    return patch


def select_best_patches(folder_path):
    discs = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
    disc_patches = {disc: [] for disc in discs}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz') and '_seg' not in filename:
            for disc in discs:
                if f"{disc}_patch" in filename:
                    file_path = os.path.join(folder_path, filename)
                    img = nib.load(file_path)
                    resolution = img.header.get_zooms()
                    voxel_volume = resolution[0] * resolution[1] * resolution[2]
                    
                    # Check for corresponding segmentation file
                    seg_filename = filename.replace('.nii.gz', '_seg.nii.gz')
                    seg_path = os.path.join(folder_path, seg_filename)
                    
                    if os.path.exists(seg_path):
                        disc_patches[disc].append((file_path, seg_path, voxel_volume))
                        
    for disc, patches in disc_patches.items():
        if len(patches) > 1:
            # Sort patches by increasing voxel volume (resolution)
            patches.sort(key=lambda x: x[2])
            
            # Keep the patch with the best resolution
            best_patch = patches[0]
            
            # Remove other patches and their segmentations
            for patch in patches[1:]:
                os.remove(patch[0])
                os.remove(patch[1])

def process_all_subjects_in_directory(root_dir, output_root_dir):
    TP_nfn = 0 
    total_nfn = 0
    TP_scs = 0
    total_scs = 0

    all_results = []

    for subject_folder in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject_folder, "anat")
        output_subject_path = os.path.join(output_root_dir, subject_folder, "anat")
        try : 
            if os.path.isdir(subject_path):
                os.makedirs(output_subject_path, exist_ok=True)
                print(subject_folder)
                TP_nfn_temp, total_nfn_temp, TP_scs_temp, total_scs_temp, df_subject = extract_patches_from_discs(subject_path, output_subject_path)
                
                TP_nfn += TP_nfn_temp
                total_nfn += total_nfn_temp
                TP_scs += TP_scs_temp
                total_scs += total_scs_temp

                    

                all_results.append(df_subject)

        except: 
            print(f'failed for {subject_folder}')
        

    # Combine all subject data and write CSV
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv("accuracy.csv", index=False)

    return TP_nfn, total_nfn, TP_scs, total_scs

        

def main():
    # Ensure a directory argument is passed
    if len(sys.argv) != 2:
        print("Usage: python extraction.py [data_directory]")
        sys.exit(1)
    
    # Get the root directory from the command-line argument
    root_dir = sys.argv[1]
    output_dir = sys.argv[1]
    
    # Run the processing function for all subjects in the specified directory
    TP_nfn,total_nfn,TP_scs,total_scs = process_all_subjects_in_directory(root_dir, output_dir)
    print(f"True Positives for Foraminal Narrowing: {TP_nfn}")
    print(f"Total Foraminal Narrowing: {total_nfn}")
    print(f"True Positives for Spinal Canal Stenosis: {TP_scs}")
    print(f"Total Spinal Canal Stenosis: {total_scs}")
    # Calculate and print the accuracy
    if total_nfn > 0:
        accuracy_nfn = TP_nfn / total_nfn
        print(f"Accuracy for Foraminal Narrowing: {accuracy_nfn:.2f}")
    else:
        print("No foraminal narrowing cases found.")
    if total_scs > 0:
        accuracy_scs = TP_scs / total_scs
        print(f"Accuracy for Spinal Canal Stenosis: {accuracy_scs:.2f}")
    else:
        print("No spinal canal stenosis cases found.")

if __name__ == "__main__":
    main()
