import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, RandFlipd, SpatialPadd, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, Resized
)
from monai.networks.nets import DenseNet201, ResNet
import torch
from torch.utils.data import DataLoader
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
import torchio as tio
from image import Image

weight = torch.tensor([1.0, 2.0, 4.0]).cuda()


def resample(
        image,
        mm = (1.0, 1.0, 1.0),
    ):
    '''
    Resample the image to the target voxel size in mm.
    Parameters
    ----------
    image : nibabel.Nifti1Image
        The input image to resample.
    mm : tuple[float, float, float]
        The target voxel size in mm.
    Returns
    -------
    nibabel.Nifti1Image
        The resampled image.
    '''
    image_data = np.asanyarray(image.dataobj).astype(np.float64)
    # Create result
    subject = tio.Resample(mm)(tio.Subject(
        image=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine),
    ))
    output_image_data = subject.image.data.numpy()[0, ...].astype(np.float64)
    output_image = nib.Nifti1Image(output_image_data, subject.image.affine, image.header)
    # Set qform twice to ensure consistent header information
    # This addresses an issue where the second call to set_qform may alter output_image.header['pixdim']
    # Applying it here prevents inconsistencies that could arise from later image edits
    output_image.set_qform(output_image.affine)
    output_image.set_qform(output_image.affine)

    return output_image


# transformation pipeline for the data
def get_transforms():
        # Define the transform pipeline with rotation augmentation
    


    common_transforms = Compose([
        LoadImaged(keys=['image']),  # Charge l'image et la segmentation
        EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
        ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        SpatialPadd(keys=['image'], spatial_size=(8, 120, 60)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['image'], roi_size=(8, 120, 60)),  # Crop pour obtenir une taille fixe
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
        counter += 1
        #if counter> 15 :
        #    break
     
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
                    image_path = os.path.join(subject_dir, file)
                    
                    parts = image_path.split('_')
                    disk_level = f"{parts[-5]}_{parts[-4]}"

                    if os.path.exists(image_path):
                        
                        # Vérifier la forme de l'image
                        image = nib.load(image_path)

                        
                        image_data = image.get_fdata()
                        image_affine = image.affine
                        image_header = image.header

                        if image_data.ndim == 3:
                            # resample the image
                            #image_res = nib.Nifti1Image(image_data, image_affine, header=image_header)
                    
                            
                    
                            #image = resample(image_res, (4.5,0.6,0.6))
                            #image_data = image.get_fdata()

                            pixdim = image.header.get_zooms()  # Get the voxel dimensions

                            # Print out the resolution of the image
                            print(f"Subject: {subject}, File: {file}, Image Shape: {image_data.shape}, Voxel Size (mm): {pixdim}")
                                                
                        
                        
                            subject_id = (subject.replace('sub-', ''))
                            if 'left' in file:
                                label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}'
                            if 'right' in file:
                                label_column = f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                                # Flip the image along the appropriate axis (e.g., flipping along x-axis)
                                image_data = np.flip(image_data, axis=0)  # Flip along the first axis (x-axis)

                            # Obtenir l'étiquette brute
                            
                            label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                            
                            # Convertir l'étiquette textuelle en valeur numérique
                            label_numeric = text2int.get(label, -1)
                            if label_numeric != -1:
                                counter += 1
                                data.append({"image": image_path, "label": label_numeric})


    print(f"Nombre de données chargées: {counter}")
    return Dataset(data=data, transform=transform)

def train_and_evaluate_model(data_dir, csv_file, batch_size=8, lr=1e-4, epochs=12, val_split=0.25, layers= [3, 4, 6, 3]):
    # Préparer les données
    transform=get_transforms()
    data = prepare_data(data_dir, csv_file, transform)
    


    # Split train/test sets
    train_size = int((1 - val_split) * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Définir le modèle, la loss function et l'optimiseur
    
    
    model = ResNet(
            block="basic",
            layers=layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=3,
            ).cuda()

    
    
    criterion = CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Listes pour stocker la perte et l'exactitude
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    model_name = f"nfn_model_layers_{layers}_epochs_{epochs}_lr_{lr}"

    # Entraînement
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0



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


        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {correct_predictions / total_predictions}%")

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
    plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, 2e graphique
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Sauvegarder la figure complète avec les deux graphiques
    plt.tight_layout()  # Pour éviter que les graphiques se chevauchent
    plt.savefig(f'training_loss_and_confusion_matrix_{model_name}.png')
    plt.close()

    print("Graphique de la perte et matrice de confusion sauvegardés dans un seul fichier.")



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
    
    

    train_and_evaluate_model(data_dir, csv_file, batch_size=8, lr=1e-4, epochs=12, val_split=0.25, layers=[3, 4, 6, 3])
   

if __name__ == "__main__":
    main()