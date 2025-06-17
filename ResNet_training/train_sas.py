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
    ResizeWithPadOrCropd, RandLambdad, RandGaussianSharpend, Rand3DElasticd,RandBiasFieldd, Flipd, SpatialCropd
)
from monai.networks.nets import DenseNet201, ResNet
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from monai.transforms import Compose
from monai.data import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nibabel as nib
import argparse  
from augment import *
import wandb 
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description="Run MONAI script for medical image processing.")
    parser.add_argument('--data', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()

#Weights used in the loss for the challenge
weight_challenge = torch.tensor([1.0, 2.0, 4.0]).cuda()



# transformation pipeline for the data
def get_transforms(mode='basic', side='left'):
        # Define the transform pipeline with rotation augmentation
    
    first_transforms = Compose([
        LoadImaged(keys=['image']),  # Charge l'image et la segmentation
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        EnsureChannelFirstd(keys=["image"])  # S'assure que l'image et la segmentation ont la dimension de canal en premier
        ])
    
    right_flip = Compose([
        Flipd(keys=['image'], spatial_axis=0)])

    second_transforms_basic = Compose([
        SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 6)),  # crop pour récupérer la gauche
        SpatialPadd(keys=['image'], spatial_size=(60, 80, 6)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['image'], roi_size=(60, 80, 6)),  # Crop pour obtenir une taille fixe
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image']) 
        ])
    
    second_transforms_random = Compose([
        RandRotated(keys=['image'], prob=0.5, range_y=0.1),
        SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 6)),  # crop pour récupérer la gauche
        SpatialPadd(keys=['image'], spatial_size=(60, 80, 6)), 
        RandSpatialCropd(keys=['image'], roi_size=(60, 80, 6), random_size=False),  
        RandLambdad(keys=['image'],func=aug_sqrt,prob=0.05,),
        RandLambdad(keys=['image'],func=aug_sin,prob=0.05,),
        RandLambdad(keys=['image'],func=aug_exp,prob=0.05,),
        RandLambdad(keys=['image'],func=aug_sig,prob=0.05, ),
        RandLambdad(keys=['image'],func=aug_laplace,prob=0.05,),
        RandLambdad(keys=['image'],func=aug_inverse,prob=0.05, ),   
        RandBiasFieldd(keys=['image'],prob=0.05),
        RandAffined(keys=['image'],prob=0.05, padding_mode="zeros", mode=["bilinear"]), 

        RandGaussianNoised(keys=['image'], mean=0.0, std=0.1, prob=0.05),
        RandGaussianSharpend(keys=['image'], prob=0.05),   

        Rand3DElasticd(keys=['image'],prob=0.05, padding_mode="zeros", mode=["bilinear"], sigma_range=(5,7), magnitude_range=(50,150)),

        ResizeWithPadOrCropd(keys=['image'], spatial_size=(60, 80, 6)),
        RandScaleIntensityd(keys=['image'], factors=(0.8, 1.2), prob=1),  # Normalisation de l'intensité pour l'image
        ToTensord(keys=['image'])
        ])

    if mode == 'basic':
        if side == 'left':
            common_transforms = Compose([first_transforms, second_transforms_basic])
        elif side == 'right':
            common_transforms = Compose([first_transforms, right_flip, second_transforms_basic])

    elif mode == 'random':
        if side == 'left':
            common_transforms = Compose([first_transforms, second_transforms_random])
        elif side == 'right':
            common_transforms = Compose([first_transforms, right_flip, second_transforms_random])
    
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
        print(subject)
        subject_dir = os.path.join(data_dir, subject, 'anat')
        #if counter>10:
        #    break
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(image_path):
                        # Vérifier la forme de l'image
                        
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

    
def train_and_evaluate_model(device, data_dir, csv_file, batch_size=4, lr=5e-5, epochs=40, val_split=0.25, layers=[3, 4, 6, 3], wd=1e-4, augment=False):

    # Préparer les données
    data_dir_train = os.path.join(data_dir, 'training')
    data_dir_val = os.path.join(data_dir, 'validation')

    # Préparer les données
    transform_left_random=get_transforms(mode='random', side='left')
    transform_right_random=get_transforms(mode='random', side='right')
    transform_left_basic=get_transforms(mode='basic', side='left')
    transform_right_basic=get_transforms(mode='basic', side='right')
    data_left_train, data_right_train = prepare_data(data_dir_train, csv_file, transform_left=transform_left_random, transform_right=transform_right_random)
    data_left_val, data_right_val = prepare_data(data_dir_val, csv_file, transform_left=transform_left_basic, transform_right=transform_right_basic)
    
    train_dataset = ConcatDataset([data_left_train, data_right_train])
    val_dataset = ConcatDataset([data_left_val, data_right_val])

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
        'augment': augment,
        'train_set_size': len(train_dataset),
        'val_set_size': len(val_dataset),
    }
    model_name = f"sas_model_layers_{layers}_epochs_{epochs}_lr_{lr}_augmentation_{augment}_wd_{wd}"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight_challenge)
    
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
            
            inputs = batch["image"].cuda()
            labels = batch["label"].cuda()

            if counter%500 == 0 : 
                train_image= inputs[0].detach().cpu().squeeze()
                

                fig = plot_slices(image=train_image,
                            
                                    )

                wandb.log({"training images": wandb.Image(fig)})
                plt.close(fig)
            
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
                
                
                inputs = batch["image"].cuda()
                labels = batch["label"].cuda()
                #if counter%500 == 0 : 
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
                counter +=1 



        val_losses.append(val_loss / len(val_loader))
        wandb.log({"val_loss": val_losses[-1], "epoch": epoch})  # Log validation loss

        print(f"Validation Loss: {val_losses[-1]}, Validation Accuracy: {100 * correct_predictions / total_predictions}%")

        if val_losses[-1] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}. Saving model...")
            best_val_loss = val_losses[-1]
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"{model_name}.pth")

    print("Entraînement terminé.")

    
def plot_slices(image):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    

    mid_axial = image.shape[2]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(1, 6, figsize=(18, 54))
    fig.suptitle('Original Image')
    for i in range(6):
        axs[i].imshow(image[:,:,mid_axial-3+i].T, cmap='gray'); axs[i].axis('off') 


    plt.tight_layout()
    fig.show()
    
    return fig

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run MONAI script for medical image processing.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()

def main():

    # Parse command-line arguments
    args = parse_args()
    
    # Extract the data directory and CSV file path
    data_dir = args.data_dir
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
    
    train_and_evaluate_model(device, data_dir, csv_file, batch_size=8, lr=5e-5, epochs=50, val_split=0.25, layers=[3, 4, 6, 3], augment=True)
   
    wandb.finish()  



if __name__ == "__main__":
    main()