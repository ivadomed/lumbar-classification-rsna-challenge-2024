# this file aims to try different data processing and augmentation techniques to improve the model's performance
# it uses the data created by extraction_with_physical_volume.py in the preprocessing-pipeline branch

# importations 
import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, RandFlipd, SpatialPadd, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, RandRotated,
    Spacingd, RandSpatialCropd, RandBiasFieldd, CutMixd
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
import wandb
from monai.data import Dataset, DataLoader
import cut_mix_up

# weights of the loss
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

# we use patches for SCS extracted with thoses values in mm
#'RL': 60,  Right-Left 
# 'AP': 40, Anterior-Posterior
# 'SI': 30, Superior-Inferior

# the median value for the voxel size is 0.43 mm in the axial plane
# and 4.4 mm between axial planes (slice thickness)
# so basically we're having a patch of size 150*100*7 for 0.4,0.4,4.4 resampling

# transformation pipeline for the data
def get_transforms(mode='basic'):
        # Define the transform pipeline with rotation augmentation
    

    if mode == 'basic':
        common_transforms = Compose([
            LoadImaged(keys=['image']),  # Charge l'image et la segmentation
            Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
            EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
            CenterSpatialCropd(keys=['image'], roi_size=(120, 80, 6)),  # Crop pour obtenir une taille fixe
            ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
            ToTensord(keys=['image']) 
        ])
    elif mode == 'random':
        # same but changing steps as random steps
        common_transforms = Compose([
            LoadImaged(keys=['image']),  # Charge l'image et la segmentation
            Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
            EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            RandRotated(keys=['image'], prob=1, range_x=0.2),
            RandBiasFieldd(keys=['image'], prob=0.4, coeff_range=(0, 0.3)), # Random bias field
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
            RandSpatialCropd(keys=['image'], roi_size=(120, 80, 6), random_size=False),  # Crop pour obtenir une taille fixe
            ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
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
        if counter < 64:
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
    
    # Initialiser W&B avant l'entraînement
    wandb.init(project="resnet_monai_save_images", config={"epochs": 50, "batch_size": 8})
    
    # Préparer les données
    data_dir_train = os.path.join(data_dir, 'training')
    data_dir_val = os.path.join(data_dir, 'validation')

    transform=get_transforms()
    data_train = prepare_data(data_dir_train, csv_file, transform)
    data_val = prepare_data(data_dir_val, csv_file, transform)
    
    save_dir = "./saved_batches"
    os.makedirs(save_dir, exist_ok=True)  # Crée un répertoire pour les images

    # data augmentation if augment=True
    if augment:
        transform = get_transforms(mode='random')
        data_train_prime = prepare_data(data_dir_train, csv_file, transform)
        data_train_second = prepare_data(data_dir_train, csv_file, transform)
        data_train_third = prepare_data(data_dir_train, csv_file, transform)

        data_train = ConcatDataset([data_train, data_train_prime, data_train_second, data_train_third])

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
        'randbiaisfield prob and coeff': (0.4, 0.3),
        'cutmix': (0.5, 1)
    }
    model_name = f"scs_model_layers_{layers}_epochs_{epochs}_lr_{lr}_augmentation_{augment}_wd_{wd}_3times"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= wd)

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


        i = 0
        for batch in tqdm(train_loader):
            print(batch["image"].shape)

            if np.random.rand() < 0.5:
                batch = cut_mix_up.cutmixup(batch, 1)
            
            inputs = batch["image"].cuda()
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

            # saving first images
            
            # Saving first images (W&B log)
            if epoch == 0 and i < 4:  # Uniquement pour la première époque, premiers batches
                for j, img in enumerate(inputs):
                    train_image= img.detach().cpu().squeeze()
                

                    fig = plot_slices(image=train_image,
                                
                                        )

                    wandb.log({"training images": wandb.Image(fig)})
                    plt.close(fig)

            i += 1

        # Log des métriques pour chaque epoch
        wandb.log({
            "epoch_loss": running_loss / len(train_loader),
            "epoch_accuracy": correct_predictions / total_predictions,
            "epoch": epoch + 1
        })



        train_losses.append(running_loss / len(train_loader))
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

    wandb.finish()
 
def plot_slices(image):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    

    print(image.shape)
    mid_axial = image.shape[2]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    fig.suptitle('Axial Slices')

    for i in range(6):
        axs[i].imshow(image[:,:, mid_axial-3+i].T, cmap='gray')
        axs[i].axis('off')

    plt.tight_layout()
    # fig.show()
    
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

    

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=8, lr=5e-5, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], augment=False)
   

if __name__ == "__main__":
    main()