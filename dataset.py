import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom as dicom
import cv2
from tqdm import tqdm
import os
import glob
import time
import sys
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandRotate,
    RandShiftIntensity,
    ToTensor,
    RandSpatialCrop,
    LoadImage,
    SqueezeDim,
    RandRotate,
    RandSimulateLowResolution,
    Resize,
    CenterSpatialCrop,
)
import torch
from nilearn.plotting import view_img

sys.path.append(os.path.abspath(os.path.join("code")))
from image import Image
from sklearn.model_selection import train_test_split

class RSNADataset(Dataset):
    """
    Build a monai dataset, given a data folder containing volumes, a contrast and a dataframe of labels.
    The data folder must respect BIDS convention.
    
    Args
    ------
    root_dir : str, name of the data folder
    study_ids : list, list of subject ids included in the dataset
    contrast : str, name of the contrast considered. among ["T1", "T2"]
    orientation : str, name of the orientation. among ["ax", "sag"]
    label_df : pd.DataFrame, dataframe with labels of each subject
    exclude : list (optional), list of subjects to exclude    
    """
    def __init__(self, root_dir : str, study_ids : list, 
                 seqtype : str, label_df : pd.DataFrame, 
                 train : bool = True, exclude : list = None):
        
        
        orientation, contrast = seqtype.split("-")
        
        self.study_ids = []
        self.images_paths = []
        self.labels = []
        
        # Storing paths to volumes
        for i, study_id in tqdm(enumerate(study_ids)):
            
            pursuie = True
            for sub in exclude : 
                if str(study_id) in sub:
                    pursuie = False 
            
            if pursuie:
                paths = glob.glob(root_dir+"/sub-"+str(study_id)+"/anat/*"+orientation+"*"+contrast+"*.nii.gz")
                try :
                    path = paths[0]
                    label = label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int)
                    
                    # Resample minority classes
                    if train:
                        if label.sum()==0:
                            self.study_ids.append(study_id)
                            self.images_paths.append(path)                        
                            self.labels.append(label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int))
                        
                        elif label.sum()==1:
                            for i in range(10):
                                self.study_ids.append(study_id)
                                self.images_paths.append(path)                        
                                self.labels.append(label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int))
                        
                        elif label.sum()>=2:
                            for i in range(20):
                                self.study_ids.append(study_id)
                                self.images_paths.append(path)                        
                                self.labels.append(label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int))
                    
                    else:
                        self.study_ids.append(study_id)
                        self.images_paths.append(path)                        
                        self.labels.append(label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int))
                    
                except IndexError:
                    pass
        
    def __len__(self):
        return len(self.study_ids)
        
    def __getitem__(self, index):
        id = self.study_ids[index]
        volume = nib.load(self.images_paths[index]).get_fdata()
        volume = torch.Tensor(volume)
        label = self.labels[index]
        n = len(label)
        y = torch.zeros((n, 3))
        for i in range(n):
            if label[i]==0:
                y[i] = torch.tensor([1, 0, 0])
            if label[i]==1:
                y[i] = torch.tensor([0, 1, 0])
            if label[i]==2:
                y[i] = torch.tensor([0, 0, 1]) 
        
        return volume, y, id