import os
import sys

def main():
    # Ensure a directory argument is passed
    if len(sys.argv) != 3:
        print("Usage: python preprocessing.py [data_input_directory] [data_output_directory]")
        sys.exit(1)
    
    # Get the root directory from the command-line argument
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    os.system(f'python niftification.py {input_dir} ')
    os.system(f'python totalspineseg.py {input_dir}_nii')
    os.system(f'python extraction.py {input_dir}_nii {output_dir}')

    print("Preprocessing completed.")

    

if __name__ == "__main__":
    main()

