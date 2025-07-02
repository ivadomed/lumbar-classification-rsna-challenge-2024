"""
This script is a combination of niftification_challenge.py, extraction.py and totalspineseg.py.
It is used to preprocess the dataset for the RSNA challenge 2024. 

Input: 
    --data: Path to the root directory of the dataset.
    --output: Path to the output directory.
    --csv_description: Path to the CSV file with series descriptions. 
Output:
    None

Author: Thomas Dagonneau and Abel Salmona 
"""

import os
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--csv_description", type=str, required=True, help="Path to the CSV file with series descriptions.")
    args = parser.parse_args()
    return args

def main():


    args = parse_arguments()
    
    input_dir = args.data
    output_dir = args.output
    csv_name = args.csv_description
    
    os.makedirs(output_dir, exist_ok=True)
    
    os.system(f'python utils/niftification_challenge.py --data {input_dir} --output {output_dir} --csv_description {csv_name}')
    os.system(f'python utils/totalspineseg.py --data {output_dir}')
    os.system(f'python utils/extraction.py --data {output_dir}')

    print("Preprocessing completed.")

    

if __name__ == "__main__":
    main()

