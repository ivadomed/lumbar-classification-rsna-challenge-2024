import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, RandFlipd, SpatialPadd, CenterSpatialCropd, Spacingd, RandSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, Resized, RandAffined, RandGaussianNoised, RandRotated,
    ResizeWithPadOrCropd, RandLambdad, RandGaussianSharpend, Rand3DElasticd,RandBiasFieldd, Flipd
)
from monai.networks.nets import DenseNet201, ResNet
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
import argparse  
import torchio as tio
import torch.nn as nn
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--csv_file', required=True)
    parser.add_argument('--model_path', required=True)
    return parser.parse_args()



def get_transforms_sas():
    # Define the transform pipeline with rotation augmentation
    
    first_transforms = [
        LoadImaged(keys=['T2']),
        EnsureChannelFirstd(keys=['T2']),
        Spacingd(keys=['T2'], pixdim=(4, 0.4, 0.4), mode=('bilinear')),
        SpatialPadd(keys=['T2'], spatial_size=(6,100, 100)),  # Adjust padding for 2D
    ]

    second_transforms_basic = [
        CenterSpatialCropd(keys=['T2'], roi_size=(6,100, 100)),  # Adjust crop for 2D
        ScaleIntensityd(keys=['T2']), 
        NormalizeIntensityd(keys=['T2'], nonzero=True, channel_wise=True),
        ToTensord(keys=['T2'])
        ]
    
    

    
    common_transforms = Compose(first_transforms + second_transforms_basic)
       
   
        
    return common_transforms

def prepare_data(data_dir, csv_file):
    data = []
    
    labels_df = pd.read_csv(csv_file)
    counter = 0 

    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
       
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' in file and 'T2w' in file:
                    t2_path = os.path.join(subject_dir, file)
                    
                    parts = t2_path.split('_')

                    disk_level = f"{parts[-5]}_{parts[-4]}"
       

                    if os.path.exists(t2_path):
                        
                        subject_id = (subject.replace('sub-', ''))
                        if 'left' in file:
                            label_column = f'left_subarticular_stenosis_{disk_level.lower()}'

                        if 'right' in file:
                            label_column = f'right_subarticular_stenosis_{disk_level.lower()}'
                            
                        label = labels_df.loc[labels_df['study_id'] == int(subject_id), label_column].values[0]
                        # Convertir l'étiquette textuelle en valeur numérique
                        label_numeric = text2int.get(label, -1)
                        if label_numeric != -1:
                            
                            data.append({"T2": t2_path, "label": label_numeric})
                            counter +=1
                        
    print(counter)                                  
    return data


def inference_and_evaluate(device, data_dir, csv_file, model_path, batch_size=4, layers=[3, 4, 6, 3]):
    val_dir = os.path.join(data_dir, 'validation')

    # Prepare validation dataset
    val_transform = get_transforms_sas()
    val_data = prepare_data(val_dir, csv_file)
    val_dataset = Dataset(val_data, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Running Inference"):
            inputs = batch["T2"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Save CSV
    df = pd.DataFrame({
        "label": all_labels,
        "predicted": all_preds,
        "prob_normal_mild": [p[0] for p in all_probs],
        "prob_moderate": [p[1] for p in all_probs],
        "prob_severe": [p[2] for p in all_probs],
    })
    df.to_csv("predictions_val_sas.csv", index=False)
    print("Saved ensemble prediction results to predictions_val_ensemble.csv")


    # Compute weighted cross-entropy loss
    weight_tensor = torch.tensor([1.0, 2.0, 4.0], device=device)
    all_probs_tensor = torch.tensor(all_probs, device=device)
    all_labels_tensor = torch.tensor(all_labels, device=device)

    weighted_loss = F.cross_entropy(all_probs_tensor, all_labels_tensor, weight=weight_tensor)
    print(f"Weighted Cross-Entropy Loss: {weighted_loss.item():.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Ensemble)\nWeighted CE Loss = {weighted_loss.item():.4f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val_ensemble.png", bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    data_dir = args.data
    csv_file = args.csv_file
    model_path = args.model_path  # Add model path argument

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    if not os.path.exists(data_dir) or not os.path.exists(csv_file):
        print("Invalid data path or CSV file.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference_and_evaluate(
        device=device,
        data_dir=data_dir,
        csv_file=csv_file,
        model_path=model_path,
        batch_size=2,
        layers=[3, 4, 6, 3]
    )



if __name__ == "__main__":
    main()