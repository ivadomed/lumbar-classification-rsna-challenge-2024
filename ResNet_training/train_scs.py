"""
This script is used to train a ResNet model for the RSNA lumbar classification challenge 2024 for the spinal canal stenosis pathology.
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
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, RandFlipd, SpatialPadd, CenterSpatialCropd, Spacingd, RandSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, Resized, RandAffined, RandGaussianNoised, RandRotated,
    ResizeWithPadOrCropd, RandLambdad, RandGaussianSharpend, Rand3DElasticd,RandBiasFieldd, Flipd
)
from monai.networks.nets import DenseNet201, ResNet
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
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
from monai.data import Dataset, DataLoader
import wandb
import pytorch_lightning as pl
from augment import *

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run MONAI script for medical image processing.")
    parser.add_argument('--data', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()


# weights of the loss
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()


# transformation pipeline for the data
def get_transforms(mode='basic'):
        # Define the transform pipeline with rotation augmentation
    

    if mode == 'basic':
        common_transforms = Compose([
            LoadImaged(keys=['image']),  # Charge l'image et la segmentation
            EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
            CenterSpatialCropd(keys=['image'], roi_size=(120, 80, 6)),  # Crop pour obtenir une taille fixe
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
            ToTensord(keys=['image']) 
        ])
    elif mode == 'random':
        # same but changing steps as random steps
        common_transforms = Compose([
            LoadImaged(keys=['image']),  # Charge l'image et la segmentation
            EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
            RandRotated(keys=['image'], prob=0.5, range_y=0.1),
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)), 
            RandSpatialCropd(keys=['image'], roi_size=(120, 80, 6), random_size=False),  
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

            #Rand3DElasticd(keys=['image'],prob=0.05, padding_mode="zeros", mode=["bilinear"], sigma_range=(5,7), magnitude_range=(50,150)),

            ResizeWithPadOrCropd(keys=['image'], spatial_size=(120, 80, 6)),
            RandScaleIntensityd(keys=['image'], factors=(0.8, 1.2), prob=1),  # Normalisation de l'intensité pour l'image
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
                        
                        label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                        
                        # Convertir l'étiquette textuelle en valeur numérique
                        label_numeric = text2int.get(label, -1)
                        if label_numeric != -1:
                            counter += 1
                            data.append({"image": image_path, "label": label_numeric})


    print(f"Nombre de données chargées: {counter}")
    return Dataset(data=data, transform=transform)

def train_and_evaluate_model(device, data_dir, csv_file, batch_size=4, lr=1e-4, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], wd=0.0001, augment=False):
    # Préparer les données
    data_dir_train = os.path.join(data_dir, 'training')
    data_dir_val = os.path.join(data_dir, 'validation')

    transform_basic=get_transforms()
    transform_random=get_transforms(mode='random')
    data_train = prepare_data(data_dir_train, csv_file, transform_random)
    data_val = prepare_data(data_dir_val, csv_file, transform_basic)
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    
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
        'train_set_size': len(data_train),
        'val_set_size': len(data_val),
        'randbiaisfield prob and coeff': (0.4, 0.3)
    }
    model_name = f"scs_model_layers_{layers}_epochs_{epochs}_lr_{lr}_augmentation_{augment}_wd_{wd}_3times"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay= wd)

    # Listes pour stocker la perte et l'exactitude
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 

    # Entraînement
    for epoch in range(epochs):

        counter = 0 
        
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0



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



def main():

    # Parse command-line arguments
    args = parse_args()
    
    # Extract the data directory and CSV file path
    data_dir = args.data
    csv_file = args.csv_file
    

    config = None
    output_path = "output_path"
    wandb.init(project=f'ResNet_scs', config=config, save_code=True, dir=output_path)


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
    
   # Specify the GPU index (0, 1, 2, ...)
    gpu_id = 0  # Change this to the desired GPU index
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=8, lr=5e-5, epochs=40, val_split=0.25, layers=[3, 4, 6, 3], augment=True)
   
    wandb.finish()  


if __name__ == "__main__":
    main()