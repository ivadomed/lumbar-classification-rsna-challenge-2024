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



def get_transforms_scs():
    # Define the transform pipeline with rotation augmentation
    
    common_transforms = Compose([
        LoadImaged(keys=['image']),  # Charge l'image et la segmentation
        EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['image'], roi_size=(120, 80, 6)),  # Crop pour obtenir une taille fixe
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image']) 
    ])
    
    return common_transforms

def prepare_data(data_dir, csv_file, transform):
    data = []
    labels_df = pd.read_csv(csv_file)
    
    counter = 0

    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
        print(subject)
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(image_path):
                      
                        subject_id = (subject.replace('sub-', ''))
                        
                        label_column = f'spinal_canal_stenosis_{disk_level.lower()}'
                        # Obtenir l'étiquette brute
                        
                        label = labels_df.loc[labels_df['study_id'] == int(subject_id), label_column].values[0]
                        
                        # Convertir l'étiquette textuelle en valeur numérique
                        label_numeric = text2int.get(label, -1)
                        if label_numeric != -1:
                            counter += 1
                            data.append({"image": image_path, "label": label_numeric})


    print(f"Nombre de données chargées: {counter}")
    return Dataset(data=data, transform=transform)

def inference_and_evaluate(device, data_dir, csv_file, model_path, batch_size=4, layers=[3, 4, 6, 3]):
    val_dir = os.path.join(data_dir, 'validation')

    # Prepare validation dataset
    val_transform = get_transforms_scs()
    val_dataset = prepare_data(val_dir, csv_file, val_transform)
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

    weight = torch.tensor([1.0, 2.0, 4.0], device=device)
    criterion = CrossEntropyLoss(weight=weight)
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Running Inference"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

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
    df.to_csv("predictions_val_scs.csv", index=False)
    print("Saved ensemble prediction results to predictions_val_scs.csv")


    weighted_loss = val_loss / len(val_loader)
    print(f"Weighted Cross-Entropy Loss: {weighted_loss:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Normal/Mild', 'Moderate', 'Severe']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix SCS\nWeighted CE Loss = {weighted_loss:.4f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val_scs.png", bbox_inches='tight')
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