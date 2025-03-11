import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import monai
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
from image import Image
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from augment import *

weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

from monai.transforms import Transform
from monai.transforms import MapTransform




"""class ExtractMiddleSliceD(MapTransform):
    def __call__(self, data):
        d = dict(data)
    
        for key in self.keys:
            #print(key)
           
            # Print the shape before extracting the middle slice
            #print(f"Before extraction - {key}: {d[key].shape}")
            # Extract the middle slice along the depth dimension
            middle_slice = d[key].shape[1] // 2
            d[key] = d[key][:, middle_slice, :, :]
            # Print the shape after extracting the middle slice
            #print(f"After extraction - {key}: {d[key].shape}")
        return d"""



class ExtractMiddleSlicesMosaicD(MapTransform):
    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            # Extract the 4 middle slices along the depth dimension
            depth = d[key].shape[1]
            start_slice = (depth // 2) - 1
            middle_slices = d[key][:, start_slice:start_slice + 4, :, :]

            # Create a mosaic of the 4 slices
            # Assuming the slices are of shape (C, D, H, W)
            # We will arrange them in a 2x2 grid
            slice1, slice2, slice3, slice4 = middle_slices[:, 0, :, :], middle_slices[:, 1, :, :], middle_slices[:, 2, :, :], middle_slices[:, 3, :, :]
            top_row = np.concatenate((slice1, slice2), axis=2)
            bottom_row = np.concatenate((slice3, slice4), axis=2)
            mosaic = np.concatenate((top_row, bottom_row), axis=1)

            d[key] = mosaic

        return d



def plot_slices(image):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    

    mid_sagittal = image.shape[1]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(2, 1, figsize=(10, 20))
    fig.suptitle('Original Image')
    
    axs[0].imshow(image[0,:,:].T, cmap='gray'); axs[0].axis('off') 
    axs[1].imshow(image[1,:,:].T, cmap='gray'); axs[1].axis('off') 
        
  
    
    plt.tight_layout()
    fig.show()
    
    return fig



def get_transforms(mode='basic', side='left'):
        # Define the transform pipeline with rotation augmentation
    
    first_transforms = Compose([
        LoadImaged(keys=['T1','T2']),
        EnsureChannelFirstd(keys=['T1','T2']),
        SpatialPadd(keys=['T1','T2'], spatial_size=(6,100, 100)),  # Adjust padding for 2D
    ])

    right_flip = Compose([
        Flipd(keys=['image'], spatial_axis=0)])

    second_transforms_basic = Compose([
        CenterSpatialCropd(keys=['T1','T2'], roi_size=(6,100, 100)),  # Adjust crop for 2D
        ExtractMiddleSlicesMosaicD(keys=['T1', 'T2']),
        ScaleIntensityd(keys=['T1','T2']),
        NormalizeIntensityd(keys=['T1','T2'], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["T1","T2"], name="combinaison"),
        ToTensord(keys=["combinaison"])
        ])
    
    second_transforms_random = Compose([
        RandRotated(keys=['T1','T2'], prob=0.5, range_y=0.1),
        SpatialPadd(keys=['T1','T2'], spatial_size=(6,100, 100)), 
        RandSpatialCropd(keys=['T1','T2'], roi_size=(6,100, 100), random_size=False),  
        RandLambdad(keys=['T1','T2'],func=aug_sqrt,prob=0.05,),
        RandLambdad(keys=['T1','T2'],func=aug_sin,prob=0.05,),
        RandLambdad(keys=['T1','T2'],func=aug_exp,prob=0.05,),
        RandLambdad(keys=['T1','T2'],func=aug_sig,prob=0.05, ),
        RandLambdad(keys=['T1','T2'],func=aug_laplace,prob=0.05,),
        RandLambdad(keys=['T1','T2'],func=aug_inverse,prob=0.05, ),   

        RandBiasFieldd(keys=['T1','T2'],prob=0.05),
        RandAffined(keys=['T1','T2'],prob=0.05, padding_mode="zeros", mode=["bilinear","bilinear"]), 

        RandGaussianNoised(keys=['T1','T2'], mean=0.0, std=0.1, prob=0.05),
        RandGaussianSharpend(keys=['T1','T2'], prob=0.05),   

        Rand3DElasticd(keys=['T1','T2'],prob=0.05, padding_mode="zeros", mode=["bilinear", "bilinear"], sigma_range=(5,7), magnitude_range=(50,150)),
        
        ResizeWithPadOrCropd(keys=['T1', 'T2'], spatial_size=(6,100, 100)),
        ExtractMiddleSlicesMosaicD(keys=['T1', 'T2']),
        ScaleIntensityd(keys=['T1','T2']),
        RandScaleIntensityd(keys=['T1','T2'], factors=(0.8, 1.2), prob=1),  # Normalisation de l'intensité pour l'image
        ConcatItemsd(keys=["T1","T2"], name="combinaison"),
        ToTensord(keys=["combinaison"])
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


"""def get_transforms(mode='basic'):
    if mode == 'basic':
        common_transforms = Compose([
            LoadImaged(keys=['T1','T2']),
            EnsureChannelFirstd(keys=['T1','T2']),
            SpatialPadd(keys=['T1','T2'], spatial_size=(6,100, 100)),  # Adjust padding for 2D
            CenterSpatialCropd(keys=['T1','T2'], roi_size=(6,100, 100)),  # Adjust crop for 2D
            ExtractMiddleSlicesMosaicD(keys=['T1', 'T2']),
            ScaleIntensityd(keys=['T1','T2']),
            NormalizeIntensityd(keys=['T1','T2'], nonzero=True, channel_wise=True),
            ConcatItemsd(keys=["T1","T2"], name="combinaison"),
            ToTensord(keys=["combinaison"])
        ])
    elif mode == 'random':
        common_transforms = Compose([
            LoadImaged(keys=['T1','T2']),
            EnsureChannelFirstd(keys=['T1','T2']),
            #ExtractMiddleSliceD(keys=['T1', 'T2']),  # Extract the middle slice
            RandRotated(keys=['T1','T2'], prob=0.5, range_y=0.1),
            SpatialPadd(keys=['T1','T2'], spatial_size=(6,100, 100)), 
            RandSpatialCropd(keys=['T1','T2'], roi_size=(6,100, 100), random_size=False),  
            RandLambdad(keys=['T1','T2'],func=aug_sqrt,prob=0.05,),
            RandLambdad(keys=['T1','T2'],func=aug_sin,prob=0.05,),
            RandLambdad(keys=['T1','T2'],func=aug_exp,prob=0.05,),
            RandLambdad(keys=['T1','T2'],func=aug_sig,prob=0.05, ),
            RandLambdad(keys=['T1','T2'],func=aug_laplace,prob=0.05,),
            RandLambdad(keys=['T1','T2'],func=aug_inverse,prob=0.05, ),   

            RandBiasFieldd(keys=['T1','T2'],prob=0.05),
            RandAffined(keys=['T1','T2'],prob=0.05, padding_mode="zeros", mode=["bilinear","bilinear"]), 

            RandGaussianNoised(keys=['T1','T2'], mean=0.0, std=0.1, prob=0.05),
            RandGaussianSharpend(keys=['T1','T2'], prob=0.05),   

            Rand3DElasticd(keys=['T1','T2'],prob=0.05, padding_mode="zeros", mode=["bilinear", "bilinear"], sigma_range=(5,7), magnitude_range=(50,150)),
            
            ResizeWithPadOrCropd(keys=['T1', 'T2'], spatial_size=(6,100, 100)),
            ExtractMiddleSlicesMosaicD(keys=['T1', 'T2']),
            ScaleIntensityd(keys=['T1','T2']),
            RandScaleIntensityd(keys=['T1','T2'], factors=(0.8, 1.2), prob=1),  # Normalisation de l'intensité pour l'image
            ConcatItemsd(keys=["T1","T2"], name="combinaison"),
            ToTensord(keys=["combinaison"])
        ])
    return common_transforms"""


    
    

def prepare_data(data_dir, csv_file, transform,train, side='left'):
    data = []
    labels_df = pd.read_csv(csv_file)
    
    counter = 0
    counter_augmented = 0
    proportions = [0,0,0]
    proportions_augmented = [0,0,0]
    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
    

        
        """if counter//2> 15 :
            break"""
       
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
                    t1_path = os.path.join(subject_dir, file)
                    
                    parts = t1_path.split('_')

                    disk_level = f"{parts[-5]}_{parts[-4]}"
              
                    for t2_file in os.listdir(subject_dir):
                        if 'right' in file and 'left' in t2_file: 
                            None 
                        elif 'left' in file and 'right' in t2_file: 
                            None
                        else :
                            if disk_level in t2_file and 'foramen' in t2_file and 'T2w' in t2_file:
                                t2_path = os.path.join(subject_dir,  t2_file)
                                
                                

                    if os.path.exists(t1_path):

                        # Vérifier la forme de l'image
                        t1_image = nib.load(t1_path)
                        t2_image = nib.load(t2_path)
                        
                        t1_image_data = t1_image.get_fdata()
                        t2_image_data = t2_image.get_fdata()

                        if t1_image_data.ndim == 3 and t2_image_data.ndim == 3 :
                            
                        
                            subject_id = (subject.replace('sub-', ''))
                            if 'left' in file:
                                label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}'
                            if 'right' in file:
                                label_column = f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                                # Flip the image along the appropriate axis (e.g., flipping along x-axis)
                                t1_image_data = np.flip(t1_image_data, axis=0)  # Flip along the first axis (x-axis)
                                t2_image_data = np.flip(t2_image_data, axis=0)

                            # Obtenir l'étiquette brute
                            
                            label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                            
                            # Convertir l'étiquette textuelle en valeur numérique
                            label_numeric = text2int.get(label, -1)
                            
                            if label_numeric != -1:
                                proportions[label_numeric] += 1 
                                counter += 1
                                proportions_augmented[label_numeric] += 1 
                                counter_augmented += 1

                                
                                
                                data.append({"T1": t1_path, "T2": t2_path, "label": label_numeric, "combinaison": None})

                                """if train and label_numeric == 2 : 
                                    for i in range (2):
                                        proportions_augmented[label_numeric] += 1 
                                        counter_augmented += 1
                                
                                        data.append({"T1": t1_path, "T2": t2_path, "label": label_numeric, "combinaison": None})"""



    print(f"Nombre de données chargées: {counter}")
    proportions = [(i/counter) for i in proportions]
    print(proportions)
    proportions_augmented = [(i/counter_augmented) for i in proportions_augmented]
    print(proportions_augmented)

    return data

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Preventing nans
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss

def train_and_evaluate_model(device, train_dir, val_dir, csv_file, batch_size=16, lr=1e-4, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], wd=1e-4, augment=False):
    # Préparer les données
    transform = get_transforms('random')
    train_data = prepare_data(train_dir, csv_file, transform, train=True)
    train_dataset = Dataset(train_data, transform)

    transform = get_transforms()
    val_dataset = prepare_data(val_dir, csv_file, transform, train=False)
    val_dataset = Dataset(val_dataset, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Définir le modèle, la loss function et l'optimiseur
    model = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=2,
        n_input_channels=2,
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
        'val_set_size': len(val_dataset)
    }
    model_name = f"nfn_t1_t2_model_layers_{layers}_epochs_{epochs}_lr_{lr}_augmentation_{augment}_wd_{wd}"

    model = model.to(device)

    # Use Focal Loss for training
    train_criterion = FocalLoss(weight=weight)
    # Use CrossEntropyLoss for validation
    val_criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Listes pour stocker la perte et l'exactitude
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    is_first = True

    # Entraînement
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        counter = 0

        for batch in tqdm(train_loader):
            inputs = batch["combinaison"].cuda()
            if counter % 5 == 0:
                train_image = inputs[0].detach().cpu().squeeze()

                fig = plot_slices(image=train_image)

                wandb.log({"training images": wandb.Image(fig)})
                plt.close(fig)

            labels = batch["label"].cuda()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = train_criterion(outputs, labels)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            counter += 1

        train_losses.append(running_loss / len(train_loader))
        accuracy = 100 * correct_predictions / total_predictions
        wandb.log({"train_loss": train_losses[-1], "accuracy": accuracy,  "epoch": epoch})  # Log training loss

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {accuracy}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs = batch["combinaison"].cuda()
                labels = batch["label"].cuda()

                # Forward pass
                outputs = model(inputs)
                loss = val_criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        val_losses.append(val_loss / len(val_loader))

        wandb.log({"val_loss": val_losses[-1], "epoch": epoch})  # Log validation loss

        print(f"Validation Loss: {val_losses[-1]}, Validation Accuracy: {100 * correct_predictions / total_predictions}%")

        if val_losses[-1] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}. Saving model...")
            best_val_loss = val_losses[-1]

            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"model/{model_name}.pth")

        wandb_logs = {
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1]
        }

        wandb_logs.clear()

    print("Entraînement terminé.")

 
    




def main():
    # Parse command-line arguments
    #args = parse_args()
    config = None
    output_path = "output_path"
    wandb.init(project=f'nfn_2D', config=config, save_code=True, dir=output_path)


    exp_logger = pl.loggers.WandbLogger(
                        name="test",
                        save_dir=output_path,
                        group="rsna-lumbar-classification",
                        log_model=True, # save best model using checkpoint callback
                        config=config)

    # Saving training script to wandb
    wandb.save(config)

    # Extract the data directory and CSV file path
    train_dir = "../../duke/public/rsna_challenge/20250212nii_data_splits/training"
    val_dir = "../../duke/public/rsna_challenge/20250212nii_data_splits/validation"
    csv_file = "../../duke/public/rsna_challenge/dcom_data/train.csv"
    


    
   # Specify the GPU index (0, 1, 2, ...)
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    

    train_and_evaluate_model(device, train_dir, val_dir, csv_file, batch_size=16, lr=1e-4, epochs=40, val_split=0.25, layers=[3, 4, 6, 3], augment=True)
   
    wandb.finish()  


if __name__ == "__main__":
    main()