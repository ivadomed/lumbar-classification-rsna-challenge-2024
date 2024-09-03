import pandas as pd
from tqdm import tqdm
import glob
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import torch
from utils import *
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImage, 
    LoadImaged,
    Spacingd,
    Orientation,
    Orientationd,
    EnsureChannelFirstd,
    EnsureChannelFirst,
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
                    
                except :
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
    ...
    """
    def __init__(self,
                 root_dir : str = None,
                 study_ids : list = None,
                 contrast : str = None,
                 description : pd.DataFrame = None,
                 coordinates : pd.DataFrame = None, 
                 labels : pd.DataFrame = None,
                 exclude : list = None,
                 transform : any = None
                 ): 
        
        self.study_ids = []
        self.images_paths = [] 
        self.coordinates = [] 
        self.labels = []
        self.instance_numbers = []
        self.transform = transform
        
        for i, study_id in tqdm(enumerate(study_ids)):
            
            # print(study_id)
            pursuie = True
            if exclude is not None:
                for sub in exclude : 
                    if str(study_id) in str(sub):
                        pursuie = False 
            
            if pursuie:
                try :
                    if contrast=="t1":
                        series_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Sagittal T1")]["series_id"].values[0]
                    elif contrast=="t2":
                        series_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Sagittal T2/STIR")]["series_id"].values[0]                      
                                           
                    X, Y = [], []
                    Z = []
                    if contrast=="t2":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Spinal Canal Stenosis")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)
                                                
                    elif contrast=="t1":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Left Neural Foraminal Narrowing")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)
                            
                    label = labels[labels["study_id"]==study_id].values[0, 1:].astype(int)
                    if label.min() < 0 or label.max() > 2 :
                        print(study_id)
                        print(label)
                    
                    if len(X)==5:
                        path = root_dir + str(study_id) + "/" + str(series_id)
                        self.study_ids.append(study_id)
                        self.images_paths.append(path) 
                        self.coordinates.append((X, Y))
                        self.instance_numbers.append(Z)
                        self.labels.append(label)
                        
                except IndexError:
                    pass
        
        return
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        label = torch.Tensor(self.labels[idx]).to(torch.long)
        study_id = self.study_ids[idx]
        
        patches = {"L1/L2": 0, "L2/L3": 0, "L3/L4": 0, "L4/L5": 0, "L5/S1": 0}
        levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        
        X, Y = self.coordinates[idx]
        Z = self.instance_numbers[idx]
        path = self.images_paths[idx]
        n = len(os.listdir(path))
        middle_slice = dicom.read_file(path + "/" + str(n // 2) + ".dcm")
        
        tmps = sorted(list(zip(X, Y)), key=lambda x : x[1])
        points = []
        for p in tmps:
            points.append(np.array(p))
            
        bbox = get_bounding_box1(points)
        
        for P in bbox:
            P = list(P)
            P.append(P[0])
            P = np.array(P)
                
        for i in range(5):
            
            theta = get_angle(bbox[i])
            s = get_sign(bbox[i])    
            img = dicom.read_file(path + "/" + str(n//2) + ".dcm")  
            rotated_img, M = rotate_image(img.pixel_array, s*theta)      
            rotated_landmarks = rotate_landmarks(bbox[i], M)
            rotated_landmarks = list(rotated_landmarks)
            rotated_landmarks.append(rotated_landmarks[0])
            rotated_landmarks = np.array(rotated_landmarks).astype(int)      
            
            i1 = np.min(rotated_landmarks[:,1])
            i2 = np.max(rotated_landmarks[:,1])
            j1 = np.min(rotated_landmarks[:,0])
            j2 = np.max(rotated_landmarks[:,0])

            patch = np.zeros((n, i2-i1, j2-j1))
            for j in range(n):
                img = dicom.read_file(path + "/" + str(j+1) + ".dcm")
                w, h = img.pixel_array.shape
                rotated_img, M = rotate_image(img.pixel_array, s*theta)
                rotated_landmarks = rotate_landmarks(bbox[i], M)
                rotated_landmarks = list(rotated_landmarks)
                rotated_landmarks.append(rotated_landmarks[0])
                rotated_landmarks = np.array(rotated_landmarks).astype(int)      
                i1 = np.min(rotated_landmarks[:,1])
                i2 = np.max(rotated_landmarks[:,1])
                j1 = np.min(rotated_landmarks[:,0])
                j2 = np.max(rotated_landmarks[:,0])
                 
            try :
                patch[j] = rotated_img[i1:i2,j1:j2]
                patches[levels[i]] = torch.Tensor(patch[None]).to(torch.float)
            except :
                print(study_id)
                print("value error")
            
        if self.transform is not None:
            patches = self.transform(patches)
        
        return patches, label, study_id

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

class UNetDataset(Dataset):
    """
    Build a monai dataset, given a data folder containing volumes, 
    a contrast and a dataframe of labels.
    The getitem method returns the volume and density maps corresponding 
    to levels region to be considered during a diagnosis.
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
    def __init__(self, 
                 root_dir : str, 
                 study_ids : list, 
                 description : pd.DataFrame,
                 coordinates : pd.DataFrame, 
                 exclude : list = None, 
                 transform : any = None):
                        
        self.transform = transform
        self.study_ids = []
        self.images_paths = [] # tuple (Sag T1, Sag T2/STIR) volume path
        self.labels = [] # Array of points of size 10 x 2
        
        # Storing paths to volumes
        for i, study_id in tqdm(enumerate(study_ids)):
            
            # print(study_id)
            pursuie = True
            if exclude is not None:
                for sub in exclude : 
                    if str(study_id) in sub:
                        pursuie = False 
            
            if pursuie:
                try :
                    t1_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Sagittal T1")]["series_id"].values[0]
                    t2_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Sagittal T2/STIR")]["series_id"].values[0]
                    
                                           
                    label_t1 = coordinates[coordinates["series_id"]==t1_id][["series_id", 
                                                                                   "study_id", 
                                                                                   "condition",
                                                                                   "instance_number", 
                                                                                   "level", 
                                                                                   "x", 
                                                                                   "y"]]
                    
                    label_t2 = coordinates[coordinates["series_id"]==t2_id][["series_id", 
                                                                                   "study_id", 
                                                                                   "condition",
                                                                                   "instance_number", 
                                                                                   "level", 
                                                                                   "x", 
                                                                                   "y"]]
                    
                    label_t1 = label_t1[label_t1["condition"]=="Left Neural Foraminal Narrowing"]
                    label_t2 = label_t2[label_t2["condition"]=="Spinal Canal Stenosis"]
                    
                    # print(study_id, id)
                    if len(label_t1)==5 and len(label_t2)==5:
                        path_t1 = root_dir + "train_images./" + str(study_id) + "/" + str(t1_id)
                        n_t1 = len(os.listdir(path_t1)) // 2
                        path_t1 += "/" + str(n_t1) + ".dcm"
                        path_t2 = root_dir + "train_images./" + str(study_id) + "/" + str(t2_id)
                        n_t2 = len(os.listdir(path_t2)) // 2
                        path_t2 += "/" + str(n_t2) + ".dcm"
                        self.labels.append((label_t1, label_t2))
                        self.study_ids.append(study_id)
                        self.images_paths.append((path_t1, path_t2)) 
                
                except IndexError:
                    pass
        
    def __len__(self):
        return len(self.study_ids)
        
    def __getitem__(self, index):
        LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        leveld = {"L1/L2": 0, "L2/L3": 1, "L3/L4": 2, "L4/L5": 3, "L5/S1": 4}
        # datad = {"image" : None, "label" : None, "study_id" : None}
        path2vol_t1, path2vol_t2 = self.images_paths[index]
        
        id = self.study_ids[index]
        
        vol_t1 = dicom.read_file(path2vol_t1)
        wt1, ht1 = vol_t1.pixel_array.shape
        vol_t2 = dicom.read_file(path2vol_t2)
        wt2, ht2 = vol_t2.pixel_array.shape
        
        compose = Compose([
            LoadImage(),
            EnsureChannelFirst(),
            # Orientation(axcodes="SP")
            # Spacing(pixdim=(1, 1, 1))
            # Resize(spatial_size=(512, 512))
        ])
        
        print("Resolution of dicom :", vol_t1.PixelSpacing)
        print(vol_t1.pixel_array.shape)
                
        vol_t1 = compose(path2vol_t1)
        vol_t2 = compose(path2vol_t2)
        
        _, H, W = vol_t1.shape
        print(H, W)
        alpha1 = [H/ht1, W/wt1]
        alpha2 = [H/ht2, W/wt2]
        
        label_t1, label_t2 = self.labels[index]
        # print(label)
        C = 5
        density_map = torch.zeros((2*C, H, W))
        
        for c in range(C):
            level = label_t1.iloc[c]["level"]
            
            x_t1 = torch.Tensor([label_t1.iloc[c]["x"]])
            y_t1 = torch.Tensor([label_t1.iloc[c]["y"]])
            
            x_t2 = torch.Tensor([label_t2.iloc[c]["x"]])
            y_t2 = torch.Tensor([label_t2.iloc[c]["y"]])
            
            print(x_t1, y_t1)
            
            X = torch.arange(H).view(1, H, 1)
            Y = torch.arange(W).view(1, 1, W)             
                      
            X_t1 = (X- alpha1[0]*x_t1) / H**.5
            Y_t1= (Y-(W-alpha1[1]*y_t1)) / W**.5
            
            X_t2 = (X - alpha2[0]*x_t2) / H**.5
            Y_t2= (Y-(W - alpha2[1]*y_t2)) / W**.5

            level = label_t1.iloc[c]["level"]
            k = leveld[level]
            density_map[k] = (torch.exp(-X_t1**2 - Y_t1**2)>.5)
            level = label_t2.iloc[c]["level"]
            k = leveld[level]
            density_map[k+5] = (torch.exp(-X_t2**2 - Y_t2**2)>.5) 

        if self.transform is not None:
            vol = vol.view(1, H, W)
            vol = vol.permute(0, 2, 1)
            
            datad = {"image": vol, "label": density_map, "study_id": id, "series_id": series_id}
            # for c in range(C):
            #     datad[LEVELS[c]] = density_map[c][None]
            #     print(type(density_map[c]))
            return self.transform(datad)    
        
        datad = {"image": (vol_t1, vol_t2), 
                 "label": density_map, 
                 "study_id": id}
        
        return datad           
    
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
                         axcodes="PSL"),
            Spacingd(keys=["image"],
                     pixdim=pixdim),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )
    
    condition = seq2cond[seqtype][0].replace("_", " ").title()
    print(condition)
    data = RSNAPatchDataset(
        root_dir=folder,
        study_ids=test_id,
        seqtype=seqtype,
        cond = condition,
        coordinates=coordinates,
        label_df=id_label,
        exclude=exclude,
        transform=None,
    )
    
    print("Lenght of dataset", len(data))
    
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    
    idx = np.random.randint(len(data))
    batch_data = data.__getitem__(idx)
    print(batch_data["L1/L2"].shape) 
    loader = DataLoader(data)
    for batch_data in tqdm(loader):
        print(batch_data["L1/L2"].shape) 
        i = np.random.randint(5)
        lvl = LEVELS[i]
        c, h, w, d = batch_data[lvl].shape
        plt.figure()
        plt.imshow(batch_data[lvl][0,:,:,d//2], cmap="gray")
        plt.savefig("test.png")