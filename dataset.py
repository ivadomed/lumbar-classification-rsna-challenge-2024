import pandas as pd
from tqdm import tqdm
import glob
import torch
from monai.data import Dataset



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
                 train : bool = True, exclude : list = None,
                 transform : any = None):
                
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
        datad = {"image" : None, "label" : None, "study_id" : None}
        
        id = self.study_ids[index]
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
    
        datad["image"] = self.images_paths[index]
        datad["label"] = y 
        datad["study_id"] = id
        
        if self.transform is not None:
            return self.transform(datad)

        else:            
            return datad