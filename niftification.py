import os
import subprocess
import dcm2niix
import glob
import shutil
import nibabel as nib
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm

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
        print(f"Error converting {input_file}: {e}")
        return None

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
        for filename in filenames: 
            merged_filename = f"sub-{subject_id}_run-{series_uid}_{filename[-14:]}"
            merged_path = os.path.join(output_path, merged_filename)  # Changed to output_folder
            os.rename(filename, merged_path)
            new_paths.append(merged_path)
    elif len(filenames) == 1:
        merged_filename = f"sub-{subject_id}_run-{series_uid}.nii.gz"
        merged_path = os.path.join(output_path, merged_filename)  # Changed to output_folder
        os.rename(filenames[0], merged_path)
        new_paths.append(merged_path)
    return new_paths

# global function that processes one subject creating nifti volumes in the bids format
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
    os.makedirs(os.path.join(output_path, f'sub-{subject_id}'), exist_ok=True)
    os.makedirs(os.path.join(output_path, f'sub-{subject_id}', 'anat'), exist_ok=True)

    # process through each acquisition of the subject    
    for idx, series_uid in enumerate(ptobj['SeriesInstanceUIDs']):
        description = ptobj['SeriesDescriptions'][idx]

        convert_dicom_to_nifti(subject_id, series_uid, input_path, output_path)
        new_paths = merge_nifti_volumes(output_path, subject_id, series_uid)
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

        corrected_nifti_path = os.path.join(output_path, f"sub-{subject_id}/anat/sub-{subject_id}_acq-{acq}_rec{series_uid}_{modality}")
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

                new_path = corrected_nifti_path + base[-11:] + ext

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

# Main function to run the processing
def main():
    # Check if everything is provided
    if len(sys.argv) != 3:
        print("Usage: python niftification.py [input_folder] [csv_description]")
        sys.exit(1)

    input_folder = sys.argv[1]
    csv_description = sys.argv[2]
    output_folder = f"{input_folder}_nii"
    os.makedirs(output_folder, exist_ok=True)

    ### Create the dictionary based on the CSV file ###
    df_meta_f = pd.read_csv(csv_description)

    subject_ids = np.unique(df_meta_f["study_id"].values)

    # List out all of the Studies we have on patients.
    part_1 = os.listdir(input_folder)
    part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))

    p1 = [(x, f"{input_folder}/{x}") for x in part_1]
    meta_obj = { p[0]: { 'folder_path': p[1], 
                        'SeriesInstanceUIDs': [] 
                    } 
                for p in p1 }

    for m in meta_obj:
        meta_obj[m]['SeriesInstanceUIDs'] = list(
            filter(lambda x: x.find('.DS') == -1, 
                os.listdir(meta_obj[m]['folder_path'])
                )
        )
    # Grabs the corresponding series descriptions
    for k in tqdm(meta_obj):
        for s in meta_obj[k]['SeriesInstanceUIDs']:
            if 'SeriesDescriptions' not in meta_obj[k]:
                meta_obj[k]['SeriesDescriptions'] = []
            try:
                meta_obj[k]['SeriesDescriptions'].append(
                    df_meta_f[(df_meta_f['study_id'] == int(k)) & 
                    (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
            except:
                None

    # Process subjects and set up directories
    for subject_id in tqdm(subject_ids):  # Adjust range as needed: 1975 subjects
        subject_id = str(subject_id)
        
        # Create specific directories
        os.makedirs(os.path.join(output_folder, f'sub-{subject_id}'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, f'sub-{subject_id}', 'anat'), exist_ok=True)

        # Process subject and set up directories
        process_subject(subject_id, input_folder, output_folder, df_meta_f, meta_obj)

if __name__ == "__main__":
    main()
