import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, NormalizeIntensityd, Spacingd, ResizeWithPadOrCropd,
    SpatialCropd, SpatialPadd, CenterSpatialCropd, Flipd, Lambdad
)
from monai.networks.nets import ResNet
import torch.optim as optim
from torch.utils.data import ConcatDataset

# Weight tensor for weighted loss function (if needed)
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

# transformation pipeline for the data
def get_transforms(side='left'):
        # Define the transform pipeline with rotation augmentation
    
    first_transforms = Compose([
        LoadImaged(keys=['image']),  # Charge l'image et la segmentation
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        EnsureChannelFirstd(keys=["image"])  # S'assure que l'image et la segmentation ont la dimension de canal en premier
        ])
    
    right_flip = Compose([
        Flipd(keys=['image'], spatial_axis=0)])
    

    second_transforms_basic = Compose([
        SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 5)),  # crop pour récupérer la gauche
        SpatialPadd(keys=['image'], spatial_size=(60, 80, 5)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['image'], roi_size=(60, 80, 5)),  # Crop pour obtenir une taille fixe
        ])

    final_transforms = Compose([
        ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image'])
    ])

    if side == 'left':
        common_transforms = Compose([first_transforms, second_transforms_basic, final_transforms])
    elif side == 'right':
        common_transforms = Compose([first_transforms, right_flip, second_transforms_basic, final_transforms])

    return common_transforms

def prepare_data(data_dir, csv_file, transform_left, transform_right):
    data_right = []
    data_left = []
    labels_df = pd.read_csv(csv_file)
    
    counter = 0
    counter_invalid = 0

    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(image_path):
                        # Vérifier la forme de l'image
                        image_data = nib.load(image_path).get_fdata()
                        if image_data.ndim == 3:
                            subject_id = (subject.replace('sub-', ''))

                            label_column_sasl = f'left_subarticular_stenosis_{disk_level.lower()}'
                            label_column_sasr = f'right_subarticular_stenosis_{disk_level.lower()}'
                            # Obtenir l'étiquette brute

                            label_sasr = labels_df.loc[labels_df['study_id'] == subject_id, label_column_sasl].values[0]
                            label_sasl = labels_df.loc[labels_df['study_id'] == subject_id, label_column_sasr].values[0]
                            
                            # Convertir l'étiquette textuelle en valeur numérique
                            label_numeric_sasr = text2int.get(label_sasr, -1)
                            label_numeric_sasl = text2int.get(label_sasl, -1)
                            if label_numeric_sasr in [0, 1, 2] and label_numeric_sasl in [0, 1, 2]:
                                data_right.append({"image": image_path, "label": label_numeric_sasr})
                                data_left.append({"image": image_path, "label": label_numeric_sasl})
                                counter += 2
                            else:
                                counter_invalid += 1
                                print(f"Étiquette {label_sasr} ou {label_sasl} invalide pour {subject_id} à {disk_level}")


    print(f"Nombre de données chargées: {counter}")
    print(f"Nombre de données invalides: {counter_invalid}")
    return Dataset(data=data_left, transform=transform_left), Dataset(data=data_right, transform=transform_right)


# Inference and confusion matrix function
def inference_and_confusion_matrix(model, val_loader, device, model_weights_path):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        print(len(val_loader))
        for batch in val_loader:
            inputs = batch["image"].cuda()
            labels = batch["label"].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Moderate", "Severe"], yticklabels=["Normal", "Moderate", "Severe"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{model_weights_path}.png')

# Inference only function
def run_inference(device, val_loader, model_weights_path):
    
    # Load pre-trained model
    model = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],  # Use the same layers as during training
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model.load_state_dict(torch.load(model_weights_path))
    
    # Run inference and compute confusion matrix
    inference_and_confusion_matrix(model, val_loader, device, model_weights_path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = "../../duke/public/rsna_challenge/20250102nii_data_splits/validation"
    csv_file = "../../duke/public/rsna_challenge/dcom_data/train.csv"
    val_dataset_l, val_dataset_r = prepare_data(val_dir, csv_file, get_transforms(side='left'), get_transforms(side='right'))
    val_dataset = ConcatDataset([val_dataset_l, val_dataset_r])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    # searches for all pth files in the current directory
    for file in os.listdir('.'):
        if file.endswith('.pth'):
            model_weights_path = file
            run_inference(device, val_loader, model_weights_path)

if __name__ == "__main__":
    main()