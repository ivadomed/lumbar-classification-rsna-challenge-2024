"""
This script is used to convert the names from the balgrist dataset.

Input: 
    --data: Path to the root directory of the dataset.
    --output: Path to the output directory where the converted files will be saved.
Output:
    None

Author: Thomas Dagonneau 
"""

import os
import shutil
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory where the converted files will be saved.")
    args = parser.parse_args()
    return args

def rename_file(filename):
    if "t2_tse_sag" in filename:
        return filename.replace("t2_tse_sag", "acq-sag_T2w")
    elif "t1_tse_sag" in filename:
        return filename.replace("t1_tse_sag", "acq-sag_T1w")
    elif "t2_tse_tra" in filename:
        return filename.replace("t2_tse_tra", "acq-ax_T2w")
    else:
        return filename  # Skip files that don't match any rule

def main():
    args = parse_arguments()
    input_root = args.data
    output_root = args.output

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for subject in os.listdir(input_root):
        subject_path = os.path.join(input_root, subject)
        if not os.path.isdir(subject_path):
            continue

        output_subject_path = os.path.join(output_root, subject)
        os.makedirs(output_subject_path, exist_ok=True)

        for filename in os.listdir(subject_path):
            if filename.endswith(".nii.gz"):
                new_name = rename_file(filename)
                if new_name:
                    src = os.path.join(subject_path, filename)
                    dst = os.path.join(output_subject_path, new_name)
                    shutil.copy(src, dst)
                    print(f"Copied: {src} -> {dst}")
                else:
                    print(f"Skipping unrecognized file: {filename}")

if __name__ == "__main__":
    main()
