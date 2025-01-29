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
    ResizeWithPadOrCropd
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

weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

class SubsetAsDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        return self.subset[idx]

# transformation pipeline for the data
def get_transforms(mode='basic'):
        # Define the transform pipeline with rotation augmentation
    

    if mode == 'basic':
        common_transforms = Compose([
            LoadImaged(keys=['T1','T2']),  # Charge l'image et la segmentation
            Spacingd(keys=['T1','T2'], pixdim=(2.4, 0.6, 0.6), mode=('bilinear')),  # Ré-échantillonnage de l'image
            EnsureChannelFirstd(keys=['T1','T2']),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            SpatialPadd(keys=['T1','T2'], spatial_size=(10, 70, 70)),  # Padding pour atteindre une taille fixe
            CenterSpatialCropd(keys=['T1','T2'], roi_size=(10, 70, 70)),  # Crop pour obtenir une taille fixe
            ScaleIntensityd(keys=['T1','T2']),  # Normalisation de l'intensité pour l'image
            NormalizeIntensityd(keys=['T1','T2'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
            ConcatItemsd(keys=["T1","T2"], name="combinaison"),
            ToTensord(keys=["combinaison"]) 
        ])
    elif mode == 'random':
        # same but changing steps as random steps
        common_transforms = Compose([
            LoadImaged(keys=['T1','T2']),  # Charge l'image et la segmentation
            Spacingd(keys=['T1','T2'], pixdim=(2.4, 0.6, 0.6), mode=('bilinear')),  # Ré-échantillonnage de l'image
            EnsureChannelFirstd(keys=['T1','T2']),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            RandRotated(keys=['T1','T2'], prob=1, range_y=0.1),  # Rotation aléatoire
            SpatialPadd(keys=['T1','T2'], spatial_size=(10, 70, 70)),  # Padding pour atteindre une taille fixe
            RandSpatialCropd(keys=['T1','T2'], roi_size=(10, 70, 70), random_size=False),  # Crop pour obtenir une taille fixe
            ResizeWithPadOrCropd(keys=['T1', 'T2'], spatial_size=(10, 70, 70)),
            RandScaleIntensityd(keys=['T1','T2'], factors=(0.8, 1.2), prob=1),  # Normalisation de l'intensité pour l'image
            NormalizeIntensityd(keys=['T1','T2'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
            ConcatItemsd(keys=["T1","T2"], name="combinaison"),
            ToTensord(keys=["combinaison"]) 
        ])
    return common_transforms
    
    
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
    fig, axs = plt.subplots(2, 6, figsize=(10, 30))
    fig.suptitle('Original Image')
    for i in range(6):
        axs[0, i].imshow(image[0,mid_sagittal-3+i,:,:].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(image[1,mid_sagittal-3+i,:,:].T, cmap='gray'); axs[1, i].axis('off') 
        
  
    
    plt.tight_layout()
    fig.show()
    
    return fig


def prepare_data(data_dir, csv_file, transform, side='left'):
    data = []
    labels_df = pd.read_csv(csv_file)
    
    counter = 0
    proportions = [0,0,0]
    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
        
        if counter//3> 15 :
            break
       
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
                                
                                data.append({"T1": t1_path, "T2": t2_path, "label": label_numeric, "combinaison": None})


    print(f"Nombre de données chargées: {counter}")
    """proportions = [1/(i/counter) for i in proportions]
    print(proportions)
    """
    print(data)
    return Dataset(data=data, transform=transform)

def train_and_evaluate_model(device, train_dir, val_dir, csv_file, batch_size=4, lr=1e-4, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], wd=1e-4, augment=False):
    # Préparer les données
    transform=get_transforms()
    train_dataset = prepare_data(train_dir, csv_file, transform)
    val_dataset = prepare_data(val_dir, csv_file, transform)
    # constant key for random gen
    seed = 42
    generator = torch.Generator().manual_seed(seed)

     

    # data augmentation if augment=True
    for i in range(1):
        transform=get_transforms('random')
        train_aug = prepare_data(train_dir, csv_file, transform)
        # data_aug_prime = prepare_data(data_dir, csv_file, transform)
        
        # then turn the subset for training back into a dataset
        """train_dataset = SubsetAsDataset(train_dataset)
        train_aug = SubsetAsDataset(train_aug)
        """#train_aug_prime = SubsetAsDataset(train_aug_prime)

        # then concatenate the two datasets
        train_dataset = ConcatDataset([train_dataset, train_aug])
        # train_dataset = ConcatDataset([train_dataset, train_aug, train_aug_prime])

        

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Définir le modèle, la loss function et l'optimiseur
    model = ResNet(
            block="bottleneck",
            layers=layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
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
    
    criterion = CrossEntropyLoss(weight=weight)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= wd)

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
            if counter%5 == 0 : 
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {100 * correct_predictions / total_predictions}%")

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
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)


        val_losses.append(val_loss / len(val_loader))
        print(f"Validation Loss: {val_losses[-1]}, Validation Accuracy: {100 * correct_predictions / total_predictions}%")

        

        if val_losses[-1] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}. Saving model...")
            best_val_loss = val_losses[-1]
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"{model_name}.pth")

        wandb_logs = {
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1]
                
            }
    
        wandb_logs.clear()

        

    print("Entraînement terminé.")


   # saving a plot of the training and its results
    plt.figure(figsize=(15, 7))

   

    # Deuxième sous-graphe : Graphique de la perte d'entraînement et validation
    #plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, 2e graphique
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Ajouter les hyperparamètres sur le graphique
    hyperparams_text = '\n'.join([f"{key}: {value}" for key, value in hyperparameters.items()])
    plt.text(0.02, 0.98, f"Hyperparameters:\n{hyperparams_text}", 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Ajouter la meilleure validation loss sur le graphique
    plt.text(0.98, 0.02, f"Best Validation Loss: {best_val_loss:.4f}", 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', 
            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


    # Sauvegarder la figure complète avec les deux graphiques
    plt.tight_layout()  # Pour éviter que les graphiques se chevauchent
    plt.savefig(f'training_loss_{model_name}.png')
    plt.close()
 
    


# Function to parse command-line arguments
"""def parse_args():
    parser = argparse.ArgumentParser(description="Run MONAI script for medical image processing.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()"""

def main():
    # Parse command-line arguments
    #args = parse_args()
    config = None
    output_path = "output_path"
    wandb.init(project=f'nfn', config=config, save_code=True, dir=output_path)


    exp_logger = pl.loggers.WandbLogger(
                        name="test",
                        save_dir=output_path,
                        group="rsna-lumbar-classification",
                        log_model=True, # save best model using checkpoint callback
                        config=config)

    # Saving training script to wandb
    wandb.save(config)

    # Extract the data directory and CSV file path
    train_dir = "../../duke/public/rsna_challenge/20250102nii_data_splits/training"
    val_dir = "../../duke/public/rsna_challenge/20250102nii_data_splits/validation"
    csv_file = "../../duke/public/rsna_challenge/dcom_data/train.csv"
    


    
   # Specify the GPU index (0, 1, 2, ...)
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    

    train_and_evaluate_model(device, train_dir, val_dir, csv_file, batch_size=4, lr=1e-4, epochs=60, val_split=0.25, layers=[3, 4, 6, 3], augment=True)
   
    wandb.finish()  


if __name__ == "__main__":
    main()