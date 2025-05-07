import os
import sys

def main():
    # Ensure a directory argument is passed
    if len(sys.argv) != 4:
        print("Usage: python preprocessing.py [data_input_directory] [data_output_directory]")
        sys.exit(1)
    
    # Get the root directory from the command-line argument
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    csv_name = sys.argv[3]
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    os.system(f'python preprocessing/niftification.py {input_dir} {output_dir} {csv_name}')
    os.system(f'python preprocessing/totalspineseg.py {output_dir}')
    os.system(f'python preprocessing/extraction.py {output_dir}')

    print("Preprocessing completed.")

    

if __name__ == "__main__":
    main()

