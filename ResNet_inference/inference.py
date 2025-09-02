from inference_nfn import inference_nfn
from inference_scs import inference_scs 
from inference_sas import inference_sas 
import csv 
import torch
import os
import argparse 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_csv', required=True)
    return parser.parse_args()

def main():

    print("Starting inference...")

    args = parse_args()
    data_dir = args.data
    model_path = args.model_path  
    output_csv = args.output_csv

    if not os.path.exists(model_path):
        print(f"Error: Model folder not found at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_nfn = inference_nfn(
            device=device,
            data_dir=data_dir,
            model_path=model_path,
        )

    pred_sas = inference_sas(
            device=device,
            data_dir=data_dir,
            model_path=model_path,
        )


    pred_scs = inference_scs(
            device=device,
            data_dir=data_dir,
            model_path=model_path,
        )
    
    pred = pred_nfn + pred_sas + pred_scs

    pred_sorted = sorted(pred, key=lambda x: x[0])

    

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["subject", "pathology", "level", "Normal/Mild", "Moderate", "Severe"])
        
        # Write each prediction
        for label, output in pred_sorted:
            parts = label.split("_")
            subject = parts[0]  # e.g. sub-121
            pathology = "_".join(parts[1:-2])  # e.g. left_neural_foraminal_narrowing
            level = "_".join(parts[-2:])  # e.g. l5_s1
            
            writer.writerow([subject, pathology, level, round(output[0], 2), round(output[1], 2), round(output[2], 2)])

            
            
if __name__ == "__main__":
    main()