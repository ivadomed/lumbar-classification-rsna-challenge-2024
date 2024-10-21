## STEP 1 ##

# this is the first part of the preprocessing pipe line
# it aims to convert the raw data to nifti format and save it with a bids format

import os
import subprocess
import dcm2niix
import glob
import shutil
import nibabel as nib
import numpy as np


# use a subprocess to convert the dicom images to nifti format, requires the output path
def convert_dicom_to_nifti(subject_id, series_uid, input_path, output_path):
    """
    Convert DICOM images to NIfTI format using dcm2niix.
    
    Parameters:
    subject_id (str): The subject identifier.
    series_uid (str): The series instance UID.
    input_path (str): Path to the DICOM images directory.
    output_path (str): Path to the output directory for NIfTI files.
    """
    input_file = os.path.join(input_path, subject_id, series_uid)
    output_file = os.path.join(output_path, f"{subject_id}-{series_uid}")
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    dcm2niix_command = f"dcm2niix -z y -m 2 -o {output_file} {input_file}"
    
    try:
        subprocess.run(dcm2niix_command, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        None


# do not apply this function to the axial acquisitions, as it will merge different acquisitions with different orientations
def merge_nifti_volumes(output_path, subject_id, series_uid):
    """
    Merge NIfTI volumes in the Z direction and save the merged volume if more than one volume exists.
    Otherwise, save the single volume directly.
    Rename the merged volume to the specified format.
    
    Parameters:
    output_filename (str): Output filename for the merged NIfTI volume.
    output_path (str): Path to the output directory for merged NIfTI volume.
    subject_id (str): The subject identifier.
    series_uid (str): The series instance UID.
    """
    output_path_for_merge = os.path.join(output_path, f"{subject_id}-{series_uid}")

    filenames = glob.glob(os.path.join(output_path_for_merge, '*.nii.gz'))
    filenames.sort()
    new_paths = []
    if len(filenames) > 1:
        for filename in filenames : 
            merged_filename = f"sub-{subject_id}_run-{series_uid}_{filename[-14:]}"
            merged_path = os.path.join(output_path, merged_filename)
            os.rename(filename, merged_path)
            new_paths.append(merged_path)
    elif len(filenames) == 1:
        merged_filename = f"sub-{subject_id}_run-{series_uid}.nii.gz"
        merged_path = os.path.join(output_path, merged_filename)
        os.rename(filenames[0], merged_path)
        new_paths.append(merged_path)
    return new_paths


# create the directories if they do not exist
def make_dirs_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        

# global function that process one subject creating nifti volumes in the bids format
def process_subject(subject_id, input_path, output_path, train, meta_obj):
    """
    Process DICOM to NIfTI conversion, merge volumes, and save with corrected orientation if applicable.
    
    Parameters:
    subject_id (str): The subject identifier.
    input_path (str): Path to the DICOM images directory.
    output_path (str): Path to the output directory for NIfTI files.
    train (DataFrame): DataFrame containing series information.
    meta_obj (dict): Metadata object with series information.
    """
    if subject_id not in train['study_id'].astype(str).values:
        return

    filtered_series = train[train['study_id'] == int(subject_id)].iloc[0]
    ptobj = meta_obj[str(filtered_series['study_id'])]

    if ptobj is None:
        
        return

    # create output directories if not existing
    make_dirs_if_not_exists(os.path.join('bids-rsna-lscd', f'sub-{subject_id}'))
    make_dirs_if_not_exists(os.path.join('bids-rsna-lscd', f'sub-{subject_id}', 'anat'))

    # process through each acquisition of the subject    
    for idx, series_uid in enumerate(ptobj['SeriesInstanceUIDs']):
        description = ptobj['SeriesDescriptions'][idx]

        convert_dicom_to_nifti(subject_id, series_uid, input_path, output_path)
        new_paths = merge_nifti_volumes(output_path, output_path, subject_id, series_uid)
        if 'Axial' in description and 'T2' in description:
                modality = 'T2w'
                acq = 'ax'
        elif 'Sagittal' in description and 'T1' in description:
            modality = 'T1w'
            acq = 'sag'
        elif 'Sagittal' in description and 'T2' in description:
            modality = 'T2w'
            acq = 'sag'
        else:
            continue

        corrected_nifti_path = f"bids-rsna-lscd/sub-{subject_id}/anat/sub-{subject_id}_acq-{acq}_rec{series_uid}_{modality}"
        if len(new_paths) > 1 : 
            for merged_nifti_path in new_paths : 
                anat_img = nib.load(merged_nifti_path)
                anat_data = anat_img.get_fdata()
                anat_affine = anat_img.affine
                anat_header = anat_img.header

                new_affine = np.copy(anat_affine)
                anat_header.set_qform(new_affine, code=1)
                anat_header.set_sform(new_affine, code=1)

                base, ext = os.path.splitext(merged_nifti_path)

                new_path = corrected_nifti_path+ base[-11:] + ext

                nib.save(nib.Nifti1Image(anat_data, new_affine, header=anat_header), new_path)

                
        else : 
            for merged_nifti_path in new_paths : 
                anat_img = nib.load(merged_nifti_path)
                anat_data = anat_img.get_fdata()
                anat_affine = anat_img.affine
                anat_header = anat_img.header

                new_affine = np.copy(anat_affine)
                anat_header.set_qform(new_affine, code=1)
                anat_header.set_sform(new_affine, code=1)

                nib.save(nib.Nifti1Image(anat_data, new_affine, header=anat_header), corrected_nifti_path + '.nii.gz')

                    