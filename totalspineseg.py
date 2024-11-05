## STEP 2 ##

# Second part of the preprocessing pipeline
# Applies totalspineseg to all the nii volumes and saves the segmentation in the nii volumes foldes with the extension total_seg 


import os
import shutil
import sys

def run_totalspineseg(source_dir): 
    ''' 
    This function applies totalspineseg to every scans in the source_dir and saves the segmentations in the source_dir.

    Parameters: 
    source_dir: folder to find the nii volumes organised acording to BIDS. 
    '''

    # Define temporary directories
    tss_temp_dir = "data"
    output_temp = "output_data"
    os.makedirs(tss_temp_dir, exist_ok=True)
    os.makedirs(output_temp, exist_ok=True)
    
    # Copy all the nii volumes in a shared folder to optimize the inference of TotalSpineSeg
    for subdir in os.listdir(source_dir):
        anat_path = os.path.join(source_dir, subdir, 'anat')
        if os.path.exists(anat_path):  
            for file in os.listdir(anat_path):
                file_path = os.path.join(anat_path, file)
                if os.path.isfile(file_path) and 'ax' not in file_path and 'total_seg' not in file_path:
                    shutil.copy(file_path, tss_temp_dir)
    
    # Run TotalSpineSeg segmentation
    os.system(f'totalspineseg {tss_temp_dir} {output_temp} --step1')

    # Move segmentations back into original data structure
    segmentations_into_anat(output_temp, source_dir)

    # Clean up temporary directories
    shutil.rmtree(tss_temp_dir)
    shutil.rmtree(output_temp)

def segmentations_into_anat(output_folder, nii_folder):
    '''
    Need to send the segmentations in the folder with the nii volumes. 
    '''
    
    # List of every segmentation file
    seg_folder = os.path.join(output_folder,"step1_output")
    segmentations = os.listdir(seg_folder)

    # Loop over each segmentation file
    for segmentation in segmentations:
        # Extract patient ID (everything before the first "_")
        id_patient = segmentation.split('_')[0]

        # Path to the patient's 'anat' folder
        patient_folder = os.path.join(nii_folder, id_patient, 'anat')

        # Check if 'anat' folder exists, else raise an alert
        if os.path.exists(patient_folder):
            # Source and destination paths
            source_path = os.path.join(seg_folder, segmentation)
            
            # Replace ".nii.gz" suffix with "_total_seg.nii.gz"
            modified_segmentation = segmentation.replace('.nii.gz', '_total_seg.nii.gz')

            destination_path = os.path.join(patient_folder, modified_segmentation)

            # Copy the segmentation to the correct folder
            shutil.copy(source_path, destination_path)
            
def main():
    # Ensure a directory argument is passed
    if len(sys.argv) != 2:
        print("Usage: python totalspineg.py [data_directory]")
        sys.exit(1)
    
    # Get the data directory from the command-line argument
    data_directory = sys.argv[1]
    
    # Run TotalSpineSeg processing
    run_totalspineseg(data_directory)

if __name__ == "__main__":
    main()


