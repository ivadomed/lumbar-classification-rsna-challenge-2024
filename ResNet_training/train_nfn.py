"""
This script is used to train a ResNet model for the RSNA lumbar classification challenge 2024 for the neural foraminal narrowing pathology.
Author: Thomas Dagonneau and Abel Salmona

Input: 
- data: path to the training and validation data directories
- csv_file: path to the CSV file containing dataset information

Output: 
None 

"""

import os
import torch
from tqdm import tqdm
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    ToTensord, RandFlipd, SpatialPadd, CenterSpatialCropd, Spacingd, RandSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandAffined, RandGaussianNoised, RandRotated,
    ResizeWithPadOrCropd, RandLambdad, RandGaussianSharpend, RandBiasFieldd
)
from monai.networks.nets import ResNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import argparse  
import wandb
import pytorch_lightning as pl
from utils.augment import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run script for training nfn model.")
    parser.add_argument('--data', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()


#Weights used in the loss for the challenge
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

def get_transforms(mode='basic'):
    first_transforms = [
        LoadImaged(keys=['T1']),
        EnsureChannelFirstd(keys=['T1']),
        Spacingd(keys=['T1'], pixdim=(4, 0.4, 0.4), mode=('bilinear')),
        SpatialPadd(keys=['T1'], spatial_size=(6,100, 100)),  
    ]

    second_transforms_basic = [
        CenterSpatialCropd(keys=['T1'], roi_size=(6,100, 100)),  
        ScaleIntensityd(keys=['T1']), 
        NormalizeIntensityd(keys=['T1'], nonzero=True, channel_wise=True),
        ToTensord(keys=['T1'])
        ]
    
    second_transforms_random = [
        RandFlipd(keys=['T1'], prob=0.5, spatial_axis=0),
        RandRotated(keys=['T1'], prob=0.5, range_y=0.1),
        SpatialPadd(keys=['T1'], spatial_size=(6,100, 100)), 
        RandSpatialCropd(keys=['T1'], roi_size=(6,100, 100), random_size=False),  
        RandLambdad(keys=['T1'],func=aug_sqrt,prob=0.05,),
        RandLambdad(keys=['T1'],func=aug_sin,prob=0.05,),
        RandLambdad(keys=['T1'],func=aug_exp,prob=0.05,),
        RandLambdad(keys=['T1'],func=aug_sig,prob=0.05, ),
        RandLambdad(keys=['T1'],func=aug_laplace,prob=0.05,),
        RandLambdad(keys=['T1'],func=aug_inverse,prob=0.05, ),   
        RandBiasFieldd(keys=['T1'],prob=0.05),
        RandAffined(keys=['T1'],prob=0.05, padding_mode="zeros", mode=["bilinear"]), 
        RandGaussianNoised(keys=['T1'], mean=0.0, std=0.1, prob=0.05),
        RandGaussianSharpend(keys=['T1'], prob=0.05),   
        ResizeWithPadOrCropd(keys=['T1'], spatial_size=(6, 100, 100)),
        RandScaleIntensityd(keys=['T1'], factors=(0.8, 1.2), prob=1), 
        NormalizeIntensityd(keys=['T1'], nonzero=True, channel_wise=True),  
        ToTensord(keys=['T1'])
        ]

    if mode == 'basic':
        
        common_transforms = Compose(first_transforms  + second_transforms_basic)

    elif mode == 'random':
       
        common_transforms = Compose(first_transforms + second_transforms_random)
    
    return common_transforms


def prepare_data(data_dir, csv_file):
    data = []
    labels_df = pd.read_csv(csv_file)
    counter = 0 

    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
    

        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
                    t1_path = os.path.join(subject_dir, file)
                    
                    parts = t1_path.split('_')

                    disk_level = f"{parts[-5]}_{parts[-4]}"
       

                    if os.path.exists(t1_path):
                        
                        subject_id = (subject.replace('sub-', ''))
                        
                        
                        if 'left' in file:
                            label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}'
                            
                        if 'right' in file:
                            label_column = f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                        
                        
                        label = labels_df.loc[labels_df['study_id'] == int(subject_id), label_column].values[0]
                        label_numeric = text2int.get(label, -1)
                        if label_numeric != -1:
                            
                            data.append({"T1": t1_path, "label": label_numeric})
                            counter +=1
                        
    print(counter)                                  
    return data
    
def plot_slices(image):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    

    mid_sagittal = image.shape[0]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(1, 6, figsize=(18, 54))
    fig.suptitle('Original Image')
    for i in range(6):
        axs[i].imshow(image[mid_sagittal-3+i,:,:].T, cmap='gray'); axs[i].axis('off') 
        
  
    
    plt.tight_layout()
    fig.show()
    
    return fig




def train_and_evaluate_model(device, data_dir, csv_file, batch_size=4, lr=1e-4, epochs=20, layers=[3, 4, 6, 3], wd=1e-4):

    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')
    
    train_transform=get_transforms(mode='random')
    train_data = prepare_data(train_dir, csv_file)
    train_dataset = Dataset(train_data, train_transform)

    val_transform = get_transforms(mode='basic')
    val_data = prepare_data(val_dir, csv_file)
    val_dataset = Dataset(val_data, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ResNet(
            block="bottleneck",
            layers=layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=3,
            ).cuda()
    
    model_name = f"train_nfn"
    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay= wd)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        counter = 0 

        for batch in tqdm(train_loader):
            inputs = batch["T1"].cuda()
            if counter%500 == 0 : 
                train_image= inputs[0].detach().cpu().squeeze()
                

                fig = plot_slices(image=train_image,
                            
                                    )

                wandb.log({"training images": wandb.Image(fig)})
                plt.close(fig)


            labels = batch["label"].cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            counter +=1 

        train_losses.append(running_loss / len(train_loader))

        wandb.log({"train_loss": train_losses[-1], "epoch": epoch})  
    

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {100 * correct_predictions / total_predictions}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                
                
                inputs = batch["T1"].cuda()
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
            
            torch.save(model.state_dict(), f"model/{model_name}.pth")


    print("Entraînement terminé.")



def main():
    args = parse_args()
    data_dir = args.data
    csv_file = args.csv_file

    output_path = "output_path"
    wandb.init(project=f'ResNet_nfn', save_code=True, dir=output_path)


    exp_logger = pl.loggers.WandbLogger(
                        name="test",
                        save_dir=output_path,
                        group="rsna-lumbar-classification",
                        log_model=True, # save best model using checkpoint callback
                        )

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The data directory '{data_dir}' does not exist.")
        return
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The CSV file '{csv_file}' does not exist.")
        return
    
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=2, lr=1e-4, epochs=40, layers=[3, 4, 6, 3])

    wandb.finish()  


if __name__ == "__main__":
    main()