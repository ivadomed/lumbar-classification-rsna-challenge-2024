import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, SpatialCropd, SpatialPadd, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, RandRotated,
    Spacingd, RandSpatialCropd, Flipd, RandBiasFieldd, EnsureChannelFirstd, Lambdad
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
import wandb
from wandb import Image as wandb_image

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
        ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image']) 
        ])
    
    second_transforms_random = Compose([
        RandRotated(keys=['image'], prob=1, range_x=0.2),
        SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 6)),  # crop pour récupérer la gauche
        RandBiasFieldd(keys=['image'], prob=0.4, coeff_range=(0, 0.3)), # Random bias field
        SpatialPadd(keys=['image'], spatial_size=(60, 80, 6)),  # Padding pour atteindre une taille fixe
        RandSpatialCropd(keys=['image'], roi_size=(60, 80, 6), random_size=False),  # Crop pour obtenir une taille fixe
        ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
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


def train_and_evaluate_model(device, data_dir, csv_file, batch_size=4, lr=5e-5, epochs=40, val_split=0.25, layers=[3, 4, 6, 3], wd=1e-4, augment=False):

    # Initialiser W&B avant l'entraînement
    wandb.init(project="resnet_monai_save_images", config={"epochs": 50, "batch_size": 8})

    # Préparer les données
    data_dir_train = os.path.join(data_dir, 'training')
    data_dir_val = os.path.join(data_dir, 'validation')

    # Préparer les données
    transform_left=get_transforms(mode='basic', side='left')
    transform_right=get_transforms(mode='basic', side='right')
    data_left_train, data_right_train = prepare_data(data_dir_train, csv_file, transform_left=transform_left, transform_right=transform_right)
    data_left_val, data_right_val = prepare_data(data_dir_val, csv_file, transform_left=transform_left, transform_right=transform_right)
    
    train_dataset = ConcatDataset([data_left_train, data_right_train])
    data_val = ConcatDataset([data_left_val, data_right_val])

    save_dir = "./saved_batches"
    os.makedirs(save_dir, exist_ok=True)  # Crée un répertoire pour les images

    # data augmentation if augment=True
    if augment:
        rand_trans_right = get_transforms(mode='random', side='right')
        rand_trans_left = get_transforms(mode='random', side='left')
        data_aug_left, data_aug_right = prepare_data(data_dir_train, csv_file, transform_left=rand_trans_left, transform_right=rand_trans_right)
        data_aug = ConcatDataset([data_aug_left, data_aug_right])
        data_aug_prime_left, data_aug_prime_right = prepare_data(data_dir_train, csv_file, transform_left=rand_trans_left, transform_right=rand_trans_right)
        data_aug_prime = ConcatDataset([data_aug_prime_left, data_aug_prime_right])

        # then concatenate the two datasets
        train_dataset = ConcatDataset([train_dataset, data_aug, data_aug_prime])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        'train_set_size': len(train_dataset),
        'val_set_size': len(data_val),
        'randbiaisfield prob and coeff': (0.4, 0.3)
    }
    model_name = f"sas_model_layers_{layers}_epochs_{epochs}_lr_{lr}_augmentation_{augment}_wd_{wd}"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight_challenge)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

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

        i = 0
        for batch in tqdm(train_loader):
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

            # Saving first images (W&B log)
            if epoch == 0 and i < 4:  # Uniquement pour la première époque, premiers batches
                for j, img in enumerate(inputs):
                    print(f"epoch {epoch}, batch {i}, image {j}")

                    # Convertir l'image en numpy pour W&B
                    img_numpy = img.cpu().numpy().squeeze()
                    shp = img_numpy.shape
                    mid_slice = shp[2] // 2
                    img_numpy = img_numpy[:, :, mid_slice:mid_slice+1].squeeze()

                    
                    # Log image dans W&B
                    wandb_img = wandb.Image(
                        img_numpy, 
                        caption=f"Epoch {epoch}, Batch {i}, Image {j}"
                    )
                    
                    wandb.log({
                        f"Logged_Image_{i}_{j}": wandb_img,
                        "epoch": epoch,
                        "batch": i,
                        "loss": loss.item(),
                    })

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

        if val_losses[-1] < best_val_loss*1.05:
            print("Val loss good enough, saving model")
            if val_losses[-1] < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}")
                best_val_loss = val_losses[-1]
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"{model_name}_{epoch}_{val_losses[-1]}.pth")
                       
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

    

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=8, lr=5e-5, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], augment=True)
   

if __name__ == "__main__":
    main()