import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import monai
from convnext import ConvNeXtAxial3D
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, SpatialCropd, SpatialPadd, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, RandRotated,
    Spacingd, RandSpatialCropd, Flipd, RandBiasFieldd, EnsureChannelFirstd, Lambdad, RandGaussianNoised,
    RandZoomd
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

regular_transforms = Compose([
        SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 5)),  # crop pour récupérer la gauche
        SpatialPadd(keys=['image'], spatial_size=(64, 80, 5)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['image'], roi_size=(64, 80, 5)),  # Crop pour obtenir une taille fixe
        ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=False),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image']) 
        ])


random_transforms = Compose([
        RandRotated(keys=['image'], prob=0.6, range_x=0.2),
        SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 5)),  # crop pour récupérer la gauche
        RandZoomd(keys=["image"], zoom_range=(1.0, 1.2), mode="trilinear", keep_size=True, prob=0.5),
        RandBiasFieldd(keys=['image'], prob=0.4, coeff_range=(0, 0.3)), # Random bias field
        SpatialPadd(keys=['image'], spatial_size=(64, 80, 5)),  # Padding pour atteindre une taille fixe
        RandSpatialCropd(keys=['image'], roi_size=(64, 80, 5), random_size=False),  # Crop pour obtenir une taille fixe
        RandGaussianNoised(keys=['image'], mean=0, std=0.05, prob=0.3),
        ScaleIntensityd(keys=['image'], channel_wise=False),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=False),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image'])
    ])

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
    
    if mode == 'basic':
        if side == 'left':
            common_transforms = Compose([first_transforms])
        elif side == 'right':
            common_transforms = Compose([first_transforms, right_flip])

    return common_transforms

class CustomDataset(Dataset):
    def __init__(self, data, fixed_transform=None, random_transform=None):
        self.data = [fixed_transform(d) for d in data] if fixed_transform else data
        self.random_transform = random_transform
        self.regular_transform = regular_transforms

    def __getitem__(self, index):
        data = self.data[index]
        if self.random_transform is not None:
            data = self.random_transform(data) 
        else :
            data = self.regular_transform(data)
        return data

def prepare_data(data_dir, csv_file, transform_left, transform_right, random_transforms=None):
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
    return CustomDataset(data=data_left, fixed_transform=transform_left, random_transform=random_transforms), CustomDataset(data=data_right, fixed_transform=transform_right, random_transform=random_transforms)


def train_and_evaluate_model(device, data_dir, csv_file, batch_size=16, lr=5e-5, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], epochs=80, val_split=0.25, augment=False):

    # Initialiser W&B avant l'entraînement
    wandb.init(project="convnext_sas", config={"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "val_split": val_split, "augment": augment, "depths": depths, "dims": dims})

    # Préparer les données
    data_dir_train = os.path.join(data_dir, 'training')
    data_dir_val = os.path.join(data_dir, 'validation')

    # Préparer les données
    transform_left=get_transforms(mode='basic', side='left')
    transform_right=get_transforms(mode='basic', side='right')
    data_left_train, data_right_train = prepare_data(data_dir_train, csv_file, transform_left=transform_left, transform_right=transform_right, random_transforms=random_transforms)
    data_left_val, data_right_val = prepare_data(data_dir_val, csv_file, transform_left=transform_left, transform_right=transform_right, random_transforms=None)
    
    train_dataset = ConcatDataset([data_left_train, data_right_train])
    data_val = ConcatDataset([data_left_val, data_right_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    
    # Définir le modèle, la loss function et l'optimiseur
    model = ConvNeXtAxial3D(in_chans=1, num_classes=3, depths=depths, dims=dims)

    # hyperparameters
    hyperparameters = {
        'batch_size': batch_size,
        'learning_rate': lr,
        'num_epochs': epochs,
        'val_split': val_split,
        'augment': augment,
        'depths': depths,
        'dims': dims,
        'train_set_size': len(train_dataset),
        'val_set_size': len(data_val),
        'randbiaisfield prob and coeff': (0.4, 0.3),
        'randzoom prob and range': (0.5, (1.0, 1.2)),
        'randgaussian prob and mean and std': (0.3, 0, 0.05),
        'randrotated prob and range': (0.6, 0.2),
        'dropout': 0.1
    }
    model_name = f"sas_next_3D_model_epochs_{epochs}_lr_{lr}_augmentation_{augment}_depths_{depths}_dims_{dims}, pdrop_{0.1}"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight_challenge)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Listes pour stocker la perte et l'exactitude
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 
    lrdiv = 0

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
                    train_image= img.detach().cpu().squeeze()
                    print("shape")
                    print(train_image.shape)

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

        if val_losses[-1] < best_val_loss*1.05:
            if val_losses[-1] < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}. Saving model...")
                best_val_loss = val_losses[-1]
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"{model_name}_{epoch}_val_loss_{val_losses[-1]:.4f}.pth")
            
        if lrdiv == 0 and best_val_loss < 0.65:
            lr = lr*0.6
            lrdiv = 1
        
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
 
    wandb.finish()

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
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    fig.suptitle('Axial Slices')

    for i in range(5):
        axs[i].imshow(image[:, :, mid_axial-2+i].T, cmap='gray')
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

    

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=16, lr=1e-4, epochs=100, val_split=0.25, augment=True)
   

if __name__ == "__main__":
    main()
    