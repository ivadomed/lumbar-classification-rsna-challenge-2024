import pandas as pd
from tqdm import tqdm
from image import Image
import glob
import os
import yaml
import numpy as np
import nibabel as nib
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

class SpinalCanalStenosisDataset(Dataset):
    def __init__(self, 
                 root_dir : str = None,
                 vol_paths : list = None,
                 seg_paths : list = None,
                 labels_csv : str = "./data/train.csv",
                 transform : any = None,
                 exclude : list = None):
        
        
        text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
        vol_paths.sort()
        seg_paths.sort()
        self.root_dir = root_dir
        self.vol_paths = vol_paths
        self.seg_paths = seg_paths
        self.transform = transform
        
        self.labels = pd.read_csv(labels_csv)
        self.labels = self.labels[["study_id",
                                   "spinal_canal_stenosis_l1_l2",
                                   "spinal_canal_stenosis_l2_l3",
                                   "spinal_canal_stenosis_l3_l4",
                                   "spinal_canal_stenosis_l4_l5",
                                   "spinal_canal_stenosis_l5_s1"]]
        self.labels = self.labels.replace(text2int)
        
        exclude_vol = []
        exclude_seg = []
        
        for study_id in exclude:
            for i in range(len(vol_paths)):
                if "sub-"+str(study_id) in vol_paths[i]:
                    print(study_id)
                    exclude_vol.append(vol_paths[i])
                    exclude_seg.append(seg_paths[i])
            
        for x in exclude_vol:
            vol_paths.remove(x)
            
        for x in exclude_seg:
            seg_paths.remove(x)
        
    def __len__(self):
        return len(self.vol_paths)
        
    def __getitem__(self, idx):
        
        vol_path = self.vol_paths[idx]
        x = vol_path.split("/")[-1]
        x = x[:-7]+"_0000.nii.gz"
        
        
        study_id = x.split("_")[0][4:]
        
        seg_path = self.seg_paths[idx]
        label = self.labels[self.labels["study_id"]==int(study_id)].values[0,1:].astype(int)
        if label.min() < 0 or label.max() > 2 :
            print(study_id)
        
        vol = Image(self.root_dir+"/output/input/"+x)
        vol.change_orientation("LSA")
        vol = vol.data
        seg = Image(seg_path)
        seg.change_orientation("LSA")
        seg = seg.data        
        
        D, H, W = vol.shape
        discs = np.isin(seg, [202, 203, 204, 205, 206]).astype(int)
        disc_l5 = np.isin(seg, [202]).astype(int)
        disc_l4 = np.isin(seg, [203]).astype(int)
        disc_l3 = np.isin(seg, [204]).astype(int)
        disc_l2 = np.isin(seg, [205]).astype(int)
        disc_l1 = np.isin(seg, [206]).astype(int)
        spinal_canal = np.isin(seg, [201]).astype(int)

        discs = [disc_l1, disc_l2, disc_l3, disc_l4, disc_l5]
        levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        patches = {}
        patches_seg = {}
        
        w = 20
        for i, disc in enumerate(discs):
            patch = patch_extraction(vol, disc, d=0, h=20, w=20)
            patches[levels[i]] = torch.Tensor(patch[None])
                
                
        if self.transform is not None:
            patches = self.transform(patches)
        
        return patches, label, study_id  

class ForaminalNarrowingDataset(Dataset):
    def __init__(self, 
                 root_dir : str = "../../TotalSpineSeg",
                 vol_paths : list = None,
                 seg_paths : list = None,
                 labels_csv : str = "./data/train.csv",
                 transform : any = None, 
                 exclude : list = None):
        
        
        text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
        
        self.transform = transform
        self.root_dir = root_dir
        
        self.labels = pd.read_csv(labels_csv)
        self.labels = self.labels[["study_id",
                                   "left_neural_foraminal_narrowing_l1_l2",
                                   "left_neural_foraminal_narrowing_l2_l3",
                                   "left_neural_foraminal_narrowing_l3_l4",
                                   "left_neural_foraminal_narrowing_l4_l5",
                                   "left_neural_foraminal_narrowing_l5_s1",
                                    "right_neural_foraminal_narrowing_l1_l2",
                                   "right_neural_foraminal_narrowing_l2_l3",
                                   "right_neural_foraminal_narrowing_l3_l4",
                                   "right_neural_foraminal_narrowing_l4_l5",
                                   "right_neural_foraminal_narrowing_l5_s1",
]]
        
        rows_with_nan = self.labels[self.labels.isna().any(axis=1)]
        excludes = exclude + list(rows_with_nan.values[:,0])
        self.labels = self.labels.dropna()
        self.labels = self.labels.replace(text2int)
         
        exclude_vol = []
        exclude_seg = []
        
        print(len(vol_paths))
        
        
        for study_id in excludes:
            for i in range(len(vol_paths)):
                if "sub-"+str(study_id) in vol_paths[i]:
                    print(study_id)
                    exclude_vol.append(vol_paths[i])
                    exclude_seg.append(seg_paths[i])
            
        for x in exclude_vol:
            vol_paths.remove(x)
            
        for x in exclude_seg:
            seg_paths.remove(x)
            
 
            
        print(len(vol_paths))
            
        vol_paths.sort()
        seg_paths.sort()
        self.vol_paths = vol_paths
        self.seg_paths = seg_paths
        
    def __len__(self):
        return len(self.vol_paths)
        
    def __getitem__(self, idx):
        
        vol_path = self.vol_paths[idx]
        x = vol_path.split("/")[-1]
        x = x[:-7]+"_0000.nii.gz"
        
        
        study_id = x.split("_")[0][4:]
        
        seg_path = self.seg_paths[idx]
        
        label = self.labels[self.labels["study_id"]==int(study_id)].values[0,1:].astype(int)
        
        vol = Image(self.root_dir+"/output/input/"+x)
        vol.change_orientation("LSA")
        vol = vol.data

        seg = Image(seg_path)
        seg = seg.change_orientation("LSA")
        seg = seg.data
        
        D, H, W = vol.shape
        discs = np.isin(seg, [202, 203, 204, 205, 206]).astype(int)
        disc_l5 = np.isin(seg, [202]).astype(int)
        disc_l4 = np.isin(seg, [203]).astype(int)
        disc_l3 = np.isin(seg, [204]).astype(int)
        disc_l2 = np.isin(seg, [205]).astype(int)
        disc_l1 = np.isin(seg, [206]).astype(int)
        # spinal_canal = np.isin(seg, [201]).astype(int)

        discs = [disc_l1, disc_l2, disc_l3, disc_l4, disc_l5]
        levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        patches_left, patches_right = {}, {}
        
        w = 20
        for i, disc in enumerate(discs):
            patch_l, patch_r = patch_extraction2(vol, disc, d=16, h=40, w=20)
            patches_left[levels[i]] = torch.Tensor(patch_l[None].copy())
            patches_right[levels[i]] = torch.Tensor(patch_r[None].copy())
                
        if self.transform is not None:
            patches_left = self.transform(patches_left)
            patches_right = self.transform(patches_right)
        
        return patches_left, patches_right, label, study_id

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
                 seqtype : str = None,
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
        self.seqtype = seqtype
        
        for i, study_id in tqdm(enumerate(study_ids)):
            
            # print(study_id)
            pursuie = True
            if exclude is not None:
                for sub in exclude : 
                    if str(study_id) in str(sub):
                        pursuie = False 
            
            if pursuie:
                try :
                    if seqtype.endswith("sag-T1"):
                        series_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Sagittal T1")]["series_id"].values[0]
                    elif seqtype=="sag-T2":
                        series_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Sagittal T2/STIR")]["series_id"].values[0]  
                    elif seqtype.endswith("ax-T2"):
                        series_id = description[(description["study_id"] ==study_id) & (description["series_description"]=="Axial T2")]["series_id"].values[0]                      
                                           
                    X, Y = [], []
                    Z = []
                    if seqtype=="sag-T2":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Spinal Canal Stenosis")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)
                                                
                    elif seqtype=="left-sag-T1":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Left Neural Foraminal Narrowing")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)
                            
                            
                    elif seqtype=="right-sag-T1":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Right Neural Foraminal Narrowing")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)
                            
                    elif seqtype=="left-ax-T2":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Left Subarticular Stenosis")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)
                    
                    elif seqtype=="right-ax-T2":
                        for _, _, instance_number, _, _, x, y in coordinates[(coordinates["study_id"]==study_id)&(coordinates["condition"]=="Right Subarticular Stenosis")].values:
                            X.append(x)
                            Y.append(y)
                            Z.append(instance_number)

                            
                    label = labels[labels["study_id"]==study_id].values[0, 1:].astype(int)
                    if label.min() < 0 or label.max() > 2 :
                        print(study_id)
                        print(label)
                    
                    if len(X)==5 or len(X)==10:
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
        X, Y = self.coordinates[idx]
        Z = self.instance_numbers[idx]
        path = self.images_paths[idx]
        n = len(os.listdir(path))
        
        tmps = sorted(list(zip(X, Y)), key=lambda x : x[1])
        points = []
        for p in tmps:
            points.append(np.array(p))
        
        bbox = get_bounding_box1(points)
        
        C = 5
        patches = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        if self.seqtype=="sag-T2" or self.seqtype.endswith("sag-T1"):
            bbox = get_bounding_box1(points)
            for i in range(C):    
                theta = get_angle(bbox[i])
                s = get_sign(bbox[i])    

                img = dicom.read_file(path + "/" + str(Z[i]) + ".dcm")
                # img = dicom.read_file(path + "/" + str(n//2) + ".dcm")
                rotated_img, M = rotate_image(img.pixel_array, s*theta)
                rotated_landmarks = rotate_landmarks(bbox[i], M)
                rotated_landmarks = list(rotated_landmarks)
                rotated_landmarks.append(rotated_landmarks[0])
                rotated_landmarks = np.array(rotated_landmarks).astype(int)      
                i1 = np.min(rotated_landmarks[:,1])
                i2 = np.max(rotated_landmarks[:,1])
                j1 = np.min(rotated_landmarks[:,0])
                j2 = np.max(rotated_landmarks[:,0])
                
                patch = np.zeros((3, i2-i1, j2-j1))
                for k in range(-1, 2):
                    img = dicom.read_file(path + "/" + str(Z[i]+k) + ".dcm")
                    # img = dicom.read_file(path + "/" + str(n//2) + ".dcm")
                    rotated_img, M = rotate_image(img.pixel_array, s*theta)
                    rotated_landmarks = rotate_landmarks(bbox[i], M)
                    rotated_landmarks = list(rotated_landmarks)
                    rotated_landmarks.append(rotated_landmarks[0])
                    rotated_landmarks = np.array(rotated_landmarks).astype(int)      
                    try:
                        patch[k+1] = rotated_img[i1:i2,j1:j2]
                    except:
                        print(study_id)
                try :
                    # patches[i+1] = torch.Tensor(rotated_img[i1:i2,j1:j2][None]).to(torch.float)
                    patches[i+1] = torch.Tensor(patch[None]).to(torch.float)
                except :
                    print(study_id)
                    print("value error")
        
        # elif self.seqtype==("left-sag-T1") :
        #     img = dicom.read_file(path + "/" + str(n//2) + ".dcm")
        #     bbox = get_bounding_box2(img.pixel_array, points).astype(int)

        #     for i in range(C):    
        #         # img = dicom.read_file(path + "/" + str(Z[i]) + ".dcm").pixel_array
        #         # i1 = np.min(bbox[i,:,1])
        #         # i2 = np.max(bbox[i,:,1])
        #         # j1 = np.min(bbox[i,:,0])
        #         # j2 = np.max(bbox[i,:,0])
                    
        #         # try :
        #         #     patches[i+1] = torch.Tensor(img[i1:i2,j1:j2][None]).to(torch.float)
        #         # except :
        #         #     print(study_id)
        #         #     print("value error")
                
        #         theta = get_angle(bbox[i])
        #         s = get_sign(bbox[i])    

        #         img = dicom.read_file(path + "/" + str(Z[i]) + ".dcm")
        #         rotated_img, M = rotate_image(img.pixel_array, s*theta)
        #         rotated_landmarks = rotate_landmarks(bbox[i], M)
        #         rotated_landmarks = list(rotated_landmarks)
        #         rotated_landmarks.append(rotated_landmarks[0])
        #         rotated_landmarks = np.array(rotated_landmarks).astype(int)      
                
        #         h, w = rotated_img.shape

        #         i1 = np.min(rotated_landmarks[:,1])
        #         i2 = np.max(rotated_landmarks[:,1])
        #         j1 = np.min(rotated_landmarks[:,0])
        #         j2 = np.max(rotated_landmarks[:,0])
                    
        #         try :
        #             patches[i+1] = torch.Tensor(rotated_img[i1:min(i2, h-1),j1:min(j2, w-1)][None]).to(torch.float)
        #         except :
        #             print(study_id)
        #             print("value error")
                    
        # elif self.seqtype == "right-sag-T1" :
        #     img = dicom.read_file(path + "/" + str(n//2) + ".dcm")
        #     bbox = get_bounding_box3(img.pixel_array, points).astype(int)

        #     for i in range(C):    
        #         img = dicom.read_file(path + "/" + str(Z[i]) + ".dcm").pixel_array
        #         i1 = np.min(bbox[i,:,1])
        #         i2 = np.max(bbox[i,:,1])
        #         j1 = np.min(bbox[i,:,0])
        #         j2 = np.max(bbox[i,:,0])
                    
        #         try :
        #             patches[i+1] = torch.Tensor(img[i1:i2,j1:j2][None]).to(torch.float)
        #         except :
        #             print(study_id)
        #             print("value error")
                    
        elif self.seqtype.endswith("ax-T2") :
            img = dicom.read_file(path + "/" + str(n//2) + ".dcm")
            bbox = get_bounding_box3(img.pixel_array, points, a=0.6, b=0.6).astype(int)

            for i in range(C):    
                img = dicom.read_file(path + "/" + str(Z[i]) + ".dcm")                
                patches[i+1] = torch.Tensor(img.pixel_array[None]).to(torch.float)   
            
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
    description = pd.read_csv("./data/train_series_descriptions.csv")
    condition = seq2cond[seqtype][0].replace("_", " ").title()
    print(condition)
    data = RSNAPatchDataset(
        root_dir=folder,
        study_ids=test_id,
        contrast="t2",
        description = description,
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