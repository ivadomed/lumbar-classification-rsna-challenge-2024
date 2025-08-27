"""
This script is used to train a ResNet model for the RSNA lumbar classification challenge 2024 for the subarticular stenosis pathology.
Author: Thomas Dagonneau and Abel Salmona

Input: 
- data: path to the training and validation data directories
- csv_file: path to the CSV file containing dataset information

Output: 
None 

"""


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
    ResizeWithPadOrCropd, RandLambdad, RandGaussianSharpend, Rand3DElasticd,RandBiasFieldd, Flipd, Cropd, SpatialCropd
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
from utils.image import Image
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from utils.augment import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run MONAI script for medical image processing.")
    parser.add_argument('--data', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()


#Weights used in the loss for the challenge
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

def get_transforms(mode='basic', left=True):
    # Define the transform pipeline with rotation augmentation
    
    first_transforms = [
        LoadImaged(keys=['T2']),
        EnsureChannelFirstd(keys=['T2']),
        Spacingd(keys=['T2'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        SpatialPadd(keys=['T2'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['T2'], roi_size=(120,80, 6)),  # Adjust crop for 2D


    ]

    if left: 
        crop_transform = [
            SpatialCropd(keys=["T2"], roi_size=(60, 80, 6), roi_center=(30, 40, 3))
        ]

    else: 
        crop_transform = [
            SpatialCropd(keys=["T2"], roi_size=(60, 80, 6), roi_center=(90, 40, 3))
        ]

    second_transforms_basic = [
        ScaleIntensityd(keys=['T2']), 
        NormalizeIntensityd(keys=['T2'], nonzero=True, channel_wise=True),
        ToTensord(keys=['T2'])
        ]
    
    second_transforms_random = [
        RandFlipd(keys=['T2'], prob=0.5, spatial_axis=0),
        RandRotated(keys=['T2'], prob=0.5, range_z=0.1),
        RandLambdad(keys=['T2'],func=aug_sqrt,prob=0.05,),
        RandLambdad(keys=['T2'],func=aug_sin,prob=0.05,),
        RandLambdad(keys=['T2'],func=aug_exp,prob=0.05,),
        RandLambdad(keys=['T2'],func=aug_sig,prob=0.05, ),
        RandLambdad(keys=['T2'],func=aug_laplace,prob=0.05,),
        RandLambdad(keys=['T2'],func=aug_inverse,prob=0.05, ),   
        RandBiasFieldd(keys=['T2'],prob=0.05),
        RandAffined(keys=['T2'],prob=0.05, padding_mode="zeros", mode=["bilinear"]), 

        RandGaussianNoised(keys=['T2'], mean=0.0, std=0.1, prob=0.05),
        RandGaussianSharpend(keys=['T2'], prob=0.05),   

        #Rand3DElasticd(keys=['T2'],prob=0.05, padding_mode="zeros", mode=["bilinear"], sigma_range=(5,7), magnitude_range=(50,150)),

        ResizeWithPadOrCropd(keys=['T2'], spatial_size=(60, 80, 6)),
        RandScaleIntensityd(keys=['T2'], factors=(0.8, 1.2), prob=1),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['T2'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['T2'])
        ]

    if mode == 'basic':
        common_transforms = Compose(first_transforms + crop_transform + second_transforms_basic)
       
    elif mode == 'random':
        common_transforms = Compose(first_transforms + crop_transform + second_transforms_random)
        
    return common_transforms


def prepare_data(data_dir, csv_file):
    data_left = []
    data_right = []
    labels_df = pd.read_csv(csv_file)
    counter = 0 

    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
       
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'sag' not in file and 'T2w' in file:
                    t2_path = os.path.join(subject_dir, file)
                    
                    parts = t2_path.split('_')

                    disk_level = f"{parts[-3]}_{parts[-2]}"
       

                    if os.path.exists(t2_path):
                        
                        subject_id = (subject.replace('sub-', ''))
                        
                        label_column_left = f'left_subarticular_stenosis_{disk_level.lower()}'
                        label_left = labels_df.loc[labels_df['study_id'] == int(subject_id), label_column_left].values[0]
                        # Convertir l'étiquette textuelle en valeur numérique
                        label_numeric_left = text2int.get(label_left, -1)
                        if label_numeric_left != -1:

                            data_left.append({"T2": t2_path, "label": label_numeric_left})
                            counter +=1

                    
                        label_column_right = f'right_subarticular_stenosis_{disk_level.lower()}'

                        label_right = labels_df.loc[labels_df['study_id'] == int(subject_id), label_column_right].values[0]
                        # Convertir l'étiquette textuelle en valeur numérique
                        label_numeric_right = text2int.get(label_right, -1)
                        if label_numeric_right != -1:

                            data_right.append({"T2": t2_path, "label": label_numeric_right})
                            counter +=1
                        
    print(counter)                                  
    return data_left, data_right 
    
def plot_slices(image):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to LPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    

    mid_sagittal = image.shape[2]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(1, 6, figsize=(18, 54))
    fig.suptitle('Original Image')
    for i in range(6):
        axs[i].imshow(image[:,:,mid_sagittal-3+i].T, cmap='gray'); axs[i].axis('off') 
        
  
    
    plt.tight_layout()
    fig.show()
    
    return fig




def train_and_evaluate_model(device, data_dir, csv_file, batch_size=4, lr=1e-4, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], wd=1e-4):

    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')
    # Préparer les données

    train_transform_left=get_transforms(mode='random', left=True)
    train_transform_right=get_transforms(mode='random', left=False)
    train_data_left, train_data_right = prepare_data(train_dir, csv_file)

    train_dataset_left = Dataset(train_data_left, train_transform_left)
    train_dataset_right = Dataset(train_data_right, train_transform_right)
    train_dataset = ConcatDataset([train_dataset_left,train_dataset_right])

    val_transform_left = get_transforms(mode='basic', left=True)
    val_transform_right = get_transforms(mode='basic', left=False)
    val_data_left, val_data_right = prepare_data(val_dir, csv_file)
    val_dataset_left = Dataset(val_data_left, val_transform_left)
    val_dataset_right = Dataset(val_data_right, val_transform_right)
    val_dataset = ConcatDataset([val_dataset_left, val_dataset_right])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Définir le modèle, la loss function et l'optimiseur
    model = ResNet(
            block="bottleneck",
            layers=layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=3,
            ).cuda()
    
    # hyperparameters
    hyperparameters = {
        'batch_size': batch_size,
        'learning_rate': lr,
        'num_epochs': epochs,
        'val_split': val_split,
        'layers': layers,
        'weight_decay': wd,
        'train_set_size': len(train_dataset),
        'val_set_size': len(val_dataset)
    }
    model_name = f"sas_5"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay= wd)

    # Listes pour stocker la perte et l'exactitude
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 

    # Entraînement
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        counter = 0 

        for batch in tqdm(train_loader):
            inputs = batch["T2"].cuda()
            if counter%500 == 0 : 
                train_image= inputs[0].detach().cpu().squeeze()
                

                fig = plot_slices(image=train_image,
                            
                                    )

                wandb.log({"training images": wandb.Image(fig)})
                plt.close(fig)


            labels = batch["label"].cuda()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            counter +=1 

        train_losses.append(running_loss / len(train_loader))

        wandb.log({"train_loss": train_losses[-1], "epoch": epoch})  # Log training loss
    

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {100 * correct_predictions / total_predictions}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                
                
                inputs = batch["T2"].cuda()
                labels = batch["label"].cuda()

                if counter%500 == 0 : 
                    val_image= inputs[0].detach().cpu().squeeze()
                    

                    fig = plot_slices(image=val_image ,
                                
                                        )

                    wandb.log({"val images": wandb.Image(fig)})
                    plt.close(fig)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                counter += 1 


        val_losses.append(val_loss / len(val_loader))

        accuracy = 100 * correct_predictions / total_predictions
        wandb.log({"val_loss": val_losses[-1], "epoch": epoch})  # Log validation loss

        print(f"Validation Loss: {val_losses[-1]}, Validation Accuracy: {accuracy}%")

        

        if val_losses[-1] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}. Saving model...")
            best_val_loss = val_losses[-1]
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"model/{model_name}.pth")


    print("Entraînement terminé.")



def main():
    # Parse command-line arguments
    args = parse_args()

    # Extract the data directory and CSV file path
    data_dir = args.data
    csv_file = args.csv_file


    config = None
    output_path = "output_path"
    wandb.init(project=f'ResNet_sas', config=config, save_code=True, dir=output_path)


    exp_logger = pl.loggers.WandbLogger(
                        name="test",
                        save_dir=output_path,
                        group="rsna-lumbar-classification",
                        log_model=True, # save best model using checkpoint callback
                        config=config)

    # Saving training script to wandb
    wandb.save(config)

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The data directory '{data_dir}' does not exist.")
        return
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The CSV file '{csv_file}' does not exist.")
        return
    
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=2, lr=1e-4, epochs=40, val_split=0.25, layers=[3, 4, 6, 3])

    wandb.finish()  


if __name__ == "__main__":
    main()