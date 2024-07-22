import pandas as pd
from tqdm import tqdm
import glob
import numpy as np
import torch
from monai.data import Dataset
from monai.transforms import (
    LoadImage, 
    LoadImaged,
    Compose,
    Resized,
    NormalizeIntensityd
)

class RSNADataset(Dataset):
    """
    Build a torch dataset, given a data folder containing volumes, a contrast and a dataframe of labels.
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
                 exclude : list = None, transform : any = None):
                
        orientation, contrast = seqtype.split("-")
        
        self.transform = transform
        self.study_ids = []
        self.images_paths = []
        self.labels = []
        
        # Storing paths to volumes
        for i, study_id in tqdm(enumerate(study_ids)):
            
            pursuie = True
            if exclude is not None:
                for sub in exclude : 
                    if str(study_id) in sub:
                        pursuie = False 
            
            if pursuie:
                paths = glob.glob(root_dir+"/sub-"+str(study_id)+"/anat/*"+orientation+"*"+contrast+"*.nii.gz")
                try :
                    path = paths[0]
                    self.study_ids.append(study_id)
                    self.images_paths.append(path)                        
                    label = label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int)
                    if label.min() < 0 or label.max() > 2 :
                        print(study_id)
                        print(label)
                    self.labels.append(label)
                    
                except IndexError:
                    pass
        
    def __len__(self):
        return len(self.study_ids)
        
    def __getitem__(self, index):
        datad = {"image" : None, "label" : None, "study_id" : None}
        
        id = self.study_ids[index]
        label = self.labels[index]
        # n = len(label)
        # y = torch.zeros((n, 3))
        # for i in range(n):
        #     if label[i]==0:
        #         y[i] = torch.tensor([1, 0, 0])
        #     if label[i]==1:
        #         y[i] = torch.tensor([0, 1, 0])
        #     if label[i]==2:
        #         y[i] = torch.tensor([0, 0, 1]) 
    
        datad["image"] = self.images_paths[index]
        datad["label"] = torch.Tensor(label).to(torch.long)
        datad["study_id"] = id
        
        if self.transform is not None:
            return self.transform(datad)

        else:            
            return datad
        
        
class RSNAPatchDataset(Dataset):
    """
    Build a torch dataset, given a data folder containing volumes, a contrast and a dataframe of labels.
    The data folder must respect BIDS convention.
    
    Args
    ------
    root_dir : str, name of the data folder
    study_ids : list, list of subject ids included in the dataset
    seqtype : str, name of the sequence type considered. among ["sag-T1", "sag-T2", "ax-T2"]
    cond : str, name of the condition considered
    label_df : pd.DataFrame, dataframe with labels of each subject
    coordinates : pd.DataFrame, dataframe with center coordinates for diagnosis
    exclude : list (optional), list of subjects to exclude    
    """
    def __init__(self, root_dir : str, 
                 study_ids : list, 
                 seqtype : str, 
                 cond : str,
                 label_df : pd.DataFrame, 
                 coordinates : pd.DataFrame, 
                 exclude : list = None, 
                 transform : any = None):
                
        orientation, contrast = seqtype.split("-")
        
        self.transform = transform
        self.study_ids = []
        self.images_paths = []
        self.labels = []
        self.centers = []
        
        # Storing paths to volumes
        for i, study_id in tqdm(enumerate(study_ids)):
            
            # print(study_id)
            pursuie = True
            if exclude is not None:
                for sub in exclude : 
                    if str(study_id) in sub:
                        pursuie = False 
            
            if pursuie:
                paths = glob.glob(root_dir+"/sub-"+str(study_id)+"/anat/*"+orientation+"*"+contrast+"*.nii.gz")
                try :
                    path = paths[0]
                    series_id = path.split("_")[2][3:]
                    label = label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int)
                                           
                    coor = coordinates[coordinates["series_id"]==int(series_id)][["series_id", 
                                                                                   "study_id", 
                                                                                   "condition",
                                                                                   "instance_number", 
                                                                                   "level", 
                                                                                   "x", 
                                                                                   "y"]]
                    coor = coor[coor["condition"]==cond]
                    
                    if len(coor)==5:
                        self.centers.append(coor)
                        self.labels.append(label)
                        self.study_ids.append(study_id)
                        self.images_paths.append(path) 
                        self.series_ids.append(series_id)                       
                except IndexError:
                    pass
        
    def __len__(self):
        return len(self.study_ids)
        
    def __getitem__(self, index):
        LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        leveld = {"L1/L2": 0, "L2/L3": 1, "L3/L4": 2, "L4/L5": 3, "L5/S1": 4}
        # datad = {"image" : None, "label" : None, "study_id" : None}
        path2vol = self.images_paths[index]
        id = self.study_ids[index]
        series_id = self.series_ids[index]
        label = self.labels[index]
        
        vol = LoadImage()(path2vol)
        H, W, D = vol.shape
           
        # print(label)
        C = len(label)
         
        label = self.label
        
        datad = {"label" : torch.Tensor(label).to(torch.long),
                 "L1/L2": None,
                 "L2/L3": None,
                 "L3/L4": None,
                 "L4/L5": None,
                 "L5/S1": None,
                 "study_id": id, 
                 "series_id": series_id}

        # Extract patches
        
        for c in range(C):
            level = label.iloc[c]["level"]
            
            x = torch.Tensor([label.iloc[c]["x"]])
            y = W - torch.Tensor([label.iloc[c]["y"]])
            z = D - torch.Tensor([label.iloc[c]["instance_number"]])
            k = leveld[level]
            patch = vol[max(0, x-32):min(H-1, x+32),
                        max(0, y-32):min(W-1, y+32),
                        max(0, z-4), min(D-1, z+4)]
            
            datad{LEVELS[k]} = patch
            
            
        if self.transform is not None:
            
            return self.transform(datad)    
        
        return datad