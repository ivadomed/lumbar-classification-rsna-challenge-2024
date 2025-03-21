import os
import sys
import shutil
import csv
import subprocess
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, RandRotate90d, RandFlipd, SpatialPadd, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, RandRotated,
    Spacingd, RandSpatialCropd, RandBiasFieldd, Flipd, SpatialCropd
)
from monai.networks.nets import DenseNet201, ResNet
from monai.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
from sklearn.metrics import confusion_matrix
import argparse
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops


def get_transforms_scs():
     
    common_transforms = Compose([
        LoadImaged(keys=['image']),  # Charge l'image et la segmentation
        EnsureChannelFirstd(keys=["image"]),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
        CenterSpatialCropd(keys=['image'], roi_size=(120, 80, 6)),  # Crop pour obtenir une taille fixe
        ScaleIntensityd(keys=['image']),  # Normalisation de l'intensité pour l'image
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
        ToTensord(keys=['image']) 
    ])
    
    
    return common_transforms


def prepare_data_scs(list_subjects, data_dir, transform):
    data = []
    
    counter = 0

    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in list_subjects:
        
        subject_dir = os.path.join(data_dir, f'{subject}', 'anat')
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
                            
                            
                            label_column = f'spinal_canal_stenosis_{disk_level.lower()}'
                            
                             
                            
                            counter += 1
                            label = f"{subject}_{label_column}"
                            data.append({"image": image_path, "label": label})


    print(f"Nombre de données chargées: {counter}")
    return Dataset(data=data, transform=transform)

 


def eval_scs(list_subjects, data_dir, model_scs):     
    # Préparer les données
    transform=get_transforms_scs()
    data = prepare_data_scs(list_subjects, data_dir, transform)
    data_loader = DataLoader(data, batch_size=8)

    model_scs.eval()
    
    pred = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            
            
            inputs = batch["image"].cuda()
            labels = batch["label"]
            outputs = model_scs(inputs)

            

            """image = inputs[0].detach().cpu().squeeze()
            
            mid_sagittal = image.shape[2] // 2
            fig, ax = plt.subplots(figsize=(10, 10))  # Single axis object
            fig.suptitle('scs')    
            ax.imshow(image[:, :, :mid_sagittal].T, cmap='gray')
            ax.axis('off')  
            plt.show()"""

            
            outputs = list(outputs.softmax(dim = 1).cpu().numpy())
            
            for i in range(len(labels)): 
                label = labels [i]
                output = list(outputs[i])
                pred.append((label, output))
            
    
    return pred 
 
        