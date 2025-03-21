import os
import shutil
import sys
import csv

def print_nii_gz_files(source_dir):
    """Print the file names ending with .nii.gz in the source directory."""
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                file_path =  os.path.join(root, file)
                assert os.system(f"sct_image -i {file_path} -setorient LPI -o {file_path}") ==0


def main():
    if len(sys.argv) != 2:
        print("Usage: python totalspineseg.py [data_directory]")
        sys.exit(1)

    data_directory = sys.argv[1]
    print_nii_gz_files(data_directory) 
   

if __name__ == "__main__":
    main()
