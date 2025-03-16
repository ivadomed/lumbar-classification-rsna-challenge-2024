import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import wandb
import pytorch_lightning as pl
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    ConcatItemsd, ToTensord, NormalizeIntensityd, RandScaleIntensityd,
    RandRotated, ResizeWithPadOrCropd, Spacingd
)
from monai.networks.nets import ResNet
from monai.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt


weight = torch.tensor([1.0, 4.0, 32.0]).cuda()


def prepare_data(data_dir, csv_file, transform, side='left'):
    data = []
    labels_df = pd.read_csv(csv_file)
    counter = 0
    proportions = {0: 0, 1: 0, 2: 0}
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
                    t1_path = os.path.join(subject_dir, file)
                    t2_path = os.path.join(subject_dir, file.replace('T1w', 'T2w'))
                    parts = file.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(t1_path):
                        # Vérifier la forme de l'image
                        t1_image = nib.load(t1_path)
                        t2_image = nib.load(t2_path)
                        
                        t1_image_data = t1_image.get_fdata()
                        t2_image_data = t2_image.get_fdata()

                        if t1_image_data.ndim == 3 and t2_image_data.ndim == 3:
                            subject_id = (subject.replace('sub-', ''))
                            if 'left' in file:
                                label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}'
                            if 'right' in file:
                                label_column = f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                                # Flip the image along the appropriate axis
                                t1_image_data = np.flip(t1_image_data, axis=0)
                                t2_image_data = np.flip(t2_image_data, axis=0)

                            # Obtenir l'étiquette brute
                            label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                            
                            # Convertir l'étiquette textuelle en valeur numérique
                            label_numeric = text2int.get(label, -1)
                            
                            if label_numeric != -1:
                                proportions[label_numeric] += 1
                                counter += 1
                                data.append({"T1": t1_path, "T2": t2_path, "label": label_numeric, "combinaison": None})