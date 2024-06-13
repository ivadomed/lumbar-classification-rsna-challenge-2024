import argparse
import os
import subprocess

'''
The script creates segmentations with usage of TotalSegmentator for each subject (*..nii.gz file) from the input directory.

The script takes two arguments:
    -i: Path to folder, where the subjects data is stored. The script will iterate over each subject directory.
    -o: Path to folder, where the segmentations will be stored.
    
The script will iterate over each subject directory in the input directory and for each subject directory it will 
iterate over each folder inside the subject directory.

For each nifti file in the folder, the script will create a corresponding segmentation directory structure and run the
TotalSegmentator command.

The TotalSegmentator command will segment the nifti file and save the segmentation in the corresponding segmentation
directory.

Requirements:
    - TotalSegmentator installed in a environment (https://github.com/wasserth/TotalSegmentator.git)

Example usage:
    python run_total_segmentator.py -i /path/to/subjects/data -o /path/to/output/segmentations

TODO: Try another TotalSegmentator (-> Nathan Molinier)

Authors: Katerina Krejci
Date: 2024-08-13

'''


def get_parser():
    """
    parser function
    """

    parser = argparse.ArgumentParser(
        description='The script creates segmentations with usage of TotalSegmentator for each subject (*..nii.gz file)'
                    ' from the input directory.',
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__)
    )
    parser.add_argument(
        '-i',
        required=True,
        help='Path to folder, where the subjects data is stored. The script will iterate over each subject directory. '
    )

    parser.add_argument(
        '-o',
        required=False,
        help='Path to folder, where the segmentations will be stored.'
    )

    return parser

def total_segmentator(input_dir, output_dir):
    """
    Function to run TotalSegmentator for each subject in the input directory
    :param input_dir: input directory with subjects data
    :param output_dir: output directory for segmentations
    :return:
    """
    # Iterate over each subject directory in the base directory
    for subject in os.listdir(input_dir):
        subject_path = os.path.join(input_dir, subject)

        # Check if it is a directory
        if os.path.isdir(subject_path):
            # Iterate over each folder inside the subject directory
            for folder in os.listdir(subject_path):
                folder_path = os.path.join(subject_path, folder)

                # Check if it is a directory
                if os.path.isdir(folder_path):
                    # Iterate over each nifti file in the folder
                    for file in os.listdir(folder_path):
                        if file.endswith('.nii') or file.endswith('.nii.gz'):
                            file_path = os.path.join(folder_path, file)

                            # Create the corresponding segmentation directory structure
                            seg_subject_path = os.path.join(output_dir, subject)
                            seg_folder_path = os.path.join(seg_subject_path, folder)
                            file_base_name = os.path.splitext(file)[0][
                                             :-4]  # Get the base name of the file without extension
                            seg_file_folder = os.path.join(seg_folder_path, file_base_name)

                            # Create the necessary directories
                            os.makedirs(seg_file_folder, exist_ok=True)

                            # Construct the TotalSegmentator command
                            command = [
                                'TotalSegmentator',
                                '-i', file_path,
                                '-o', seg_file_folder,
                                '--task', 'total_mr'
                            ]

                            # Run the TotalSegmentator command
                            print(f"Running TotalSegmentator for file: {file_path}")
                            subprocess.run(command, check=True)

                            print(f"Segmentation completed for file: {file_path}, saved in {seg_file_folder}")


def main():
    """
    Main function to run the TotalSegmentator for each subject in the input directory
    :return:
    """
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()
    base_dir = args.i
    segmentation_base_dir = args.o

    # Run the TotalSegmentator for each subject in the input directory
    total_segmentator(base_dir, segmentation_base_dir)

if __name__ == '__main__':
    main()

