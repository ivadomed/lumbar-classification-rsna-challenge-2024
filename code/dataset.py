import pandas as pd
from tqdm import tqdm
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, 
    LoadImaged,
    Spacingd,
    Orientationd,
    EnsureChannelFirstd,
    Compose,
    Resized,
    NormalizeIntensityd
)
from sklearn.model_selection import train_test_split

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
        self.cond = cond
        self.study_ids = []
        self.series_ids = []
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
                    label = label_df[label_df["study_id"]==study_id] #.values[0, 1:].astype(int)
                                  
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
        path2vol = self.images_paths[index]
        id = self.study_ids[index]
        series_id = self.series_ids[index]
        coor = self.centers[index]
        
        if self.transform is not None:
            
            vol = LoadImage()(path2vol)
            h, w, d = vol.shape
            vol = self.transform({"image": path2vol})
            vol = vol["image"]
            _, D, W, H = vol.shape
            alphas = np.zeros(3)
            alphas[0] = H/h
            alphas[1] = W/w
            alphas[2] = D/d
    
        else :
            vol = LoadImage()(path2vol)    
            H, W, D = vol.shape
           
        # print(label)
        C = len(coor)         
        label = self.labels[index]
        
        y = torch.zeros(5).to(int)
        levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
        for i, lvl in enumerate(levels):
            # print(id, lvl)
            # print(self.cond.lower().replace(" ", "_")+"_"+lvl)
            # print(label[self.cond.lower().replace(" ", "_")+"_"+lvl].values.astype(int))
            y[i] = label[self.cond.lower().replace(" ", "_")+"_"+lvl].values.astype(int)[0]
        
        datad = {"label" : y,
                 "L1/L2": None,
                 "L2/L3": None,
                 "L3/L4": None,
                 "L4/L5": None,
                 "L5/S1": None,
                 "study_id": id, 
                 "series_id": series_id}

        # Extract patches
        
        for c in range(C):
            level = coor.iloc[c]["level"]
            
            k = leveld[level]
            if self.transform is not None:
                x = torch.Tensor([coor.iloc[c]["x"]])
                y = torch.Tensor([coor.iloc[c]["y"]])
                z = torch.Tensor([coor.iloc[c]["instance_number"]])
                
                X, Y, Z = alphas[0]*x, alphas[1]*y, alphas[2]*z
                # print(X, Y, Z)
                # print(vol.shape)
                # print(alphas)
                patch = vol[:,
                            max(0, int(Z-8)):min(D-1, int(Z+8)),    # R/L
                            max(0, int(Y-64)):min(W-1, int(Y+64)),  # S/I
                            max(0, int(X-64)):min(H-1, int(X+64))]  # P/A
                
                datad[LEVELS[k]] = patch
                
            else :
                x = torch.Tensor([coor.iloc[c]["x"]])
                y = W - torch.Tensor([coor.iloc[c]["y"]])
                z = D - torch.Tensor([coor.iloc[c]["instance_number"]])
                patch = vol[max(0, int(x-32)):min(H-1, int(x+32)),  # A
                            max(0, int(y-32)):min(W-1, int(y+32)),  # I
                            max(0, int(z-8)):min(D-1, int(z+8))]    # L
                datad[LEVELS[k]] = patch
                
        return datad

class CustomDataset(Dataset):
    def __init__(self, root_dir : str, 
                 study_ids : list, 
                 seqtype : str, 
                 cond : str,
                 label_df : pd.DataFrame, 
                 exclude : list = None, 
                 transform : any = None):
        
        orientation, contrast = seqtype.split("-")
        
        self.transform = transform
        self.study_ids = []
        self.series_ids = []
        self.images_paths = []
        self.centers = []
        self.labels = []
                
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
                derivatives = glob.glob(root_dir+"derivatives/labels/sub-"+str(study_id)+"/anat/*"+cond.replace(" ", "")+"*.nii.gz")
                try :
                    path = paths[0]
                    series_id = path.split("_")[2][3:]
                    label = label_df[label_df["study_id"]==study_id].values[0, 1:].astype(int)
                    
                    if len(derivatives)==5: 
                        self.centers.append(derivatives)
                        self.labels.append(label)
                        self.study_ids.append(study_id)
                        self.images_paths.append(path) 
                        self.series_ids.append(series_id)                       
                except IndexError:
                    pass
        
    def __len__(self):
        return len(self.study_ids)
        
    def __getitem__(self, index: int):
        
        return

if __name__=="__main__":
    
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    
    with open("./config/config.yml", "r") as f:
        config = yaml.safe_load(f)

    folder = config["folder"]
    exclude_file = config["exclude_file"]
    
    # Sequence type
    seqtype = config["seqtype"]

    # resolution
    pixdim = config["pixdim"]
    
    # orientation
    orientation = config["orientation"]
    
    # Transform parameters
    LEVELS = config["levels"]

    # Dictionary mapping sequence type (contrast + orientation) to the associated condition.

    seq2cond = {
        "sag-T1": [
            "left_neural_foraminal_narrowing",
            "right_neural_foraminal_narrowing",
        ],
        "sag-T2": ["spinal_canal_stenosis"],
        "ax-T2": ["left_subarticular_stenosis", "right_subarticular_stenosis"],
    }

    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    id_label = pd.read_csv("./data/train.csv")
    

    cond_lev = ["study_id"]

    CONDITIONS = seq2cond[seqtype]

    for level in LEVELS:
        for cond in CONDITIONS:
            cond_lev.append(cond + "_" + level)

    # print(cond_lev)
    id_label = id_label[cond_lev]
    id_label = id_label.dropna()
    id_label = id_label.replace(text2int)
    study_ids = id_label.values[:, 0].astype(int)  # store id of each subject
    
    train_id, val_id = train_test_split(study_ids, test_size=0.3, random_state=42)
    val_id, test_id = train_test_split(val_id, test_size=0.5, random_state=42)
   
    print(study_ids)
    
    exclude = list(np.load(exclude_file))

    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], 
                         axcodes=orientation),
            Spacingd(keys=["image"],
                     pixdim=pixdim),
            # Resized(keys=["image"],
            #         spatial_size=(20, 512, 512),
            #         mode=["bilinear"]
            #),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )
    
    data = RSNAPatchDataset(
        root_dir=folder,
        study_ids=test_id,
        seqtype=seqtype,
        cond = "Left Neural Foraminal Narrowing",
        coordinates=coordinates,
        label_df=id_label,
        exclude=exclude,
        transform=None,
    )
    
    print("Lenght of dataset", len(data))
    
    idx = np.random.randint(len(data))
    batch_data = data.__getitem__(idx)
    print(batch_data["L1/L2"].shape) 
    loader = DataLoader(data)
    for batch_data in tqdm(loader):
        print(batch_data["L1/L2"].shape) 
        plt.figure()
        plt.imshow(batch_data["L5/S1"][0,:,:,4], cmap="gray")
        plt.savefig("test.png")
    