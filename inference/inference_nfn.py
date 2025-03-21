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



def get_transforms_nfn():
    common_transforms = Compose([
        LoadImaged(keys=['T1', 'T2']),  # Load the image and segmentation
        EnsureChannelFirstd(keys=['T1', 'T2']),  # Ensure the image and segmentation have the channel dimension first
        Spacingd(keys=['T1', 'T2'], pixdim=(4.0, 0.4, 0.4), mode=('bilinear', 'bilinear')),  # Resample the image
        SpatialPadd(keys=['T1', 'T2'], spatial_size=(6, 100, 100)),  # Padding to reach a fixed size
        CenterSpatialCropd(keys=['T1', 'T2'], roi_size=(6, 100, 100)),  # Crop to obtain a fixed size
        ScaleIntensityd(keys=['T1', 'T2']),  # Normalize the intensity for the image
        NormalizeIntensityd(keys=['T1', 'T2'], nonzero=True, channel_wise=True),  # Normalize the intensity on the image
        ConcatItemsd(keys=["T1", "T2"], name="combined"),
        ToTensord(keys=["combined"])
    ])

    return common_transforms


def prepare_data_nfn(list_subjects, data_dir, transform):
    data = []
    
    counter = 0
    
    # Dictionnaire de conversion des étiquettes
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in list_subjects:
        
        subject_dir = os.path.join(data_dir, f'{subject}', 'anat')
        
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
                                t2_path = os.path.join(subject_dir, t2_file)
                    
                    if os.path.exists(t1_path):
                        
                        
                            
                        
                        if 'left' in file:
                            label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}'
                        if 'right' in file:
                            label_column = f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                            

                        # Obtenir l'étiquette brute
                        
                        
                        counter += 1
                        label = f"{subject}_{label_column}"
                        data.append({"T1": t1_path, "T2": t2_path,  "label": label, "combined": None})


    print(f"Nombre de données chargées: {counter}")
    
    return Dataset(data=data, transform=transform)





def eval_nfn(list_subjects, data_dir, model_nfn): 
    
        # Préparer les données
        transform=get_transforms_nfn()
        data = prepare_data_nfn(list_subjects, data_dir, transform)
        data_loader = DataLoader(data, batch_size=4)
    
        model_nfn.eval()
        
        pred = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                
                
                inputs = batch["combined"].cuda()
                labels = batch["label"]
                
                outputs = model_nfn(inputs)
                
                outputs = list(outputs.softmax(dim = 1).cpu().numpy())
                
                for i in range(len(labels)): 
                    label = labels [i]
                    output = list(outputs[i])
                    pred.append((label, output))
        
        return pred 


