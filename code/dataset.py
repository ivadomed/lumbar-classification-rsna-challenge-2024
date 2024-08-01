import pandas as pd
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
import yaml
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
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet


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
        datad["image"] = self.images_paths[index]
        datad["label"] = torch.Tensor(label).to(torch.long)
        datad["study_id"] = id
        
        if self.transform is not None:
            return self.transform(datad)

        else:            
            return datad
        
        
class UNetDataset(Dataset):
    """
    Build a monai dataset, given a data folder containing volumes, 
    a contrast and a dataframe of labels.
    The gititem method returns the volume and density maps corresponding 
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
    def __init__(self, root_dir : str, study_ids : list, 
                 seqtype : str, coordinates : pd.DataFrame, 
                 cond : str, exclude : list = None, 
                 transform : any = None):
                
        orientation, contrast = seqtype.split("-")
        
        self.transform = transform
        self.study_ids = []
        self.series_ids = []
        self.images_paths = []
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
                try :
                    path = paths[0]
                    series_id = path.split("_")[2][3:]
                                           
                    label = coordinates[coordinates["series_id"]==int(series_id)][["series_id", 
                                                                                   "study_id", 
                                                                                   "condition",
                                                                                   "instance_number", 
                                                                                   "level", 
                                                                                   "x", 
                                                                                   "y"]]
                    label = label[label["condition"]==cond]
                    id = label["study_id"].values[0]
                    # print(study_id, id)
                    if len(label)==5:
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
        
        vol = LoadImage()(path2vol)
        H, W, D = vol.shape
         
        label = self.labels[index]
        # print(label)
        C = len(label)
        density_map = torch.zeros((C, D, H, W))
        
        for c in range(C):
            level = label.iloc[c]["level"]
            
            x = torch.Tensor([label.iloc[c]["x"]])
            y = torch.Tensor([label.iloc[c]["y"]])
            z = torch.Tensor([label.iloc[c]["instance_number"]])
            
            
            X = torch.arange(H).view(1, H, 1)
            Y = torch.arange(W).view(1, 1, W)
            Z = torch.arange(D).view(D, 1, 1)
             
                      
            Xnew = (X-x) / H**.5
            Ynew= (Y-(W-y)) / W**.5
            # dmap = torch.exp(-Xnew**2 - Ynew**2)>.5
            # print("dmap shape", dmap.repeat(D, 1, 1).shape)
            Znew = (Z-(D-z)) / D**.5
            
            k = leveld[level]
            density_map[k] = (torch.exp(-Xnew**2 - Ynew**2 - Znew**2)>.5) # dmap.repeat(D, 1, 1)
            #density_map[k] = dmap.repeat(D, 1, 1)
            # plt.imshow(density_map[k, :, :, 12]>0.5)
            # print(density_map[k, :, :, 12].max(), density_map[k, :, :, -12].max())
            # plt.savefig("center.png")
            # print(int(z))
            # plt.imshow(1e3*density_map[k, :, :, -int(z)] + vol[:,:,-int(z)])
            # plt.savefig("slice.png")
            # print(level, k, id)            
            
        if self.transform is not None:
            vol = vol.view(1, H, W, D)
            vol = vol.permute(0, 3, 1, 2)
            
            datad = {"image": vol, "label": density_map, "study_id": id, "series_id": series_id}
            # for c in range(C):
            #     datad[LEVELS[c]] = density_map[c][None]
            #     print(type(density_map[c]))
            return self.transform(datad)    
        
        datad = {"image": vol, 
                 "label": density_map, 
                 "study_id": id,
                 "series_id": series_id}
        
        return datad
    
if __name__=="__main__":
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    
    with open("config/config.yml", "r") as f:
        config = yaml.safe_load(f)

    folder = config["folder"]

    # Sequence type
    seqtype = config["seqtype"]

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

    condition = seq2cond[seqtype][0].replace("_", " ").title()
    print("Condition :", condition)
    print("Sequence type :", seqtype)

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
    
    exclude = list(np.load("data/exclude.npy"))

    transform = Compose(
        [
            Resized(keys=["image", "label"],
                    spatial_size=(20, 512, 512),
                    mode=["bilinear", "nearest"]
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )
    
    # data = UNetDataset(
    #     root_dir=folder,
    #     study_ids=test_id,
    #     seqtype=seqtype,
    #     coordinates=coordinates,
    #     cond = "Left Neural Foraminal Narrowing",
    #     exclude=exclude,
    #     transform=transform,
    # )

    data = UNetDataset(
        root_dir=folder,
        study_ids=test_id,
        seqtype=seqtype,
        coordinates=coordinates,
        cond = condition,
        exclude=exclude,
        transform=transform,
    )

    print("Lenght of dataset", len(data))

    idx = np.random.randint(len(data))
    # idx = 393
    batch_data = data.__getitem__(idx) # next(iter(data))
    vol, dmap, id = batch_data["image"], batch_data["label"], batch_data["study_id"]
    series_id = batch_data["series_id"]
    
    print("sub :", id, "idx", idx)
    print("series_id", series_id)
    print(dmap.shape)
    print(np.unique(dmap))
    p=.5
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(vol[0,8,:,:])
    idxs = []
    dmap = dmap.numpy()
    print(dmap.shape)
    for i in range(5):
        print(dmap.sum(axis=(0, 2, 3)).shape)
        idx = dmap.sum(axis=(0, 2, 3)).argmax()
        idxs.append(idx)
    ax[1].imshow(vol[0,8,:,:] + 10*(dmap[0,idxs[0],:,:]+dmap[1,idxs[1],:,:]+dmap[2,idxs[2],:,:]+dmap[3,idxs[3],:,:]+dmap[4,idxs[4],:,:]))
                
    plt.savefig("test.png")
    # gpu = config["gpu"]
    # device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    
    # model = BasicUNet(spatial_dims=3, 
    #                   in_channels=1, 
    #                   out_channels=5).to(device)

    # model.load_state_dict(torch.load("best_metric_model_unet2.pth"))
    # model.eval()
    
    # pred = model(vol[None].to(device))
    # pred = torch.sigmoid(pred)>.5
    # pred = pred.detach().cpu().numpy()
    
    # # fig, ax = plt.subplots(ncols=2)
    # # ax[0].imshow(vol[0,0,:,:])
    # plt.figure()
    
    # _, D, H, W = vol.shape
    # # vol = vol.repeat(1, D, H, W, 3) # gray
    
    # colors = [[255, 255, 178], # L1/L2
    #           [254, 204, 92],  # L2/L3
    #           [253, 141, 60],  # L3/L4
    #           [240, 59, 32],   # L4/L5
    #           [189, 0, 38]]    # L5/S1
    
        
    # _, C, D, H, W = pred.shape
    # pred_map = np.zeros((C, D, H, W, 3)).astype(int)
    # true_map = np.zeros((C, D, H, W, 3)).astype(int)
    
    # print(dmap.min(), dmap.max())
    
    # for i in range(5):
    #     for j in range(3):
    #         pred_map[i,:,:,:,j] = pred[0,i]*colors[i][j]
            
    # for i in range(5):
    #     for j in range(3):
    #         true_map[i,:,:,:,j] = dmap[i]*colors[i][j] 
    
    # fig, ax = plt.subplots(ncols=3) 
    # ax[0].imshow(vol[0,11,:,:])
    # ax[0].set_title("Random slice of volume")
    
    # ax[1].imshow(pred_map[0,5,:,:]+pred_map[1,5,:,:]+pred_map[2,5,:,:]+pred_map[3,5,:,:]+pred_map[4,5,:,:])
    # ax[1].set_title("Prediction")
    
    # print(true_map.shape)
    # idxs = []
    # for i in range(5):
    #     print(true_map.sum(axis=(0, 2, 3, 4)).shape)
    #     idx = true_map.sum(axis=(0, 2, 3, 4)).argmax()
    #     print(idx, true_map.sum(axis=(0, 2, 3, 4))[idx])
    #     idxs.append(idx)
    # ax[2].imshow(true_map[0,idxs[0],:,:]+true_map[1,idxs[1],:,:]+true_map[2,idxs[2],:,:]+true_map[3,idxs[3],:,:]
    #              +true_map[4,idxs[4],:,:])
    # ax[2].set_title("Ground truth")

        
    # plt.savefig("result.png")
    
    # print(vol.shape)
    # B, D, H, W = vol.shape
    # vol = vol.reshape((D, B, H, W))
    # print(vol.shape)
    # test = model(vol.to(device))
    # print(test.max())
    
    # coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    # study_id, series_id, instance_number, condition, level, x, y = coordinates[coordinates["study_id"]==4003253].iloc[0]

    # sagT1 = glob.glob("./bids-rsna-lscd/sub*/anat/*T2w.nii.gz")
    
    
    
    # n = 10
    # fig, ax = plt.subplots(nrows=n, figsize=(10, 100))
    # for i, path in enumerate(sagT1[:n]):
    #     img = Image(path)
    #     print(img.orientation)
        
    #     ax[i].imshow(img.data[:,:,0])
    
    # plt.savefig(f"img.png")
    
    # path = "./bids-rsna-lscd/sub-208289456/anat/sub-208289456_acq-sag_rec1491690538_T1w.nii.gz"
    # path = "./bids-rsna-lscd/sub-4003253/anat/sub-4003253_acq-sag_rec702807833_T2w.nii.gz"
    # img = transform(path)
    # print(img.orientation)
    # plt.imshow(img[:,:,-8]) # image is rotated from pi/2 degrees at the right
    # H, W, D = img.shape
    # plt.scatter(W - y, x, color="orange")
    # plt.axis("off")
    # plt.savefig(f"img.png")
    
    # series_id = path.split("_")[2][3:]
    # print(series_id)
    # print(coordinates[coordinates["series_id"]==int(series_id)][["instance_number", "level", "x", "y"]].head())
    # print(len(coordinates[coordinates["series_id"]==int(series_id)]))
    
    ## X : X first dimension
    ## Y : -Y second dimension
    ## 2 (instance number) -> -Z third dimension
    
    # sagT1 = glob.glob("./bids-rsna-lscd/sub*/anat/*T1w.nii.gz")
    # consistency = True
    # errors = []
    # for path in tqdm(sagT1):
    #     img = Image(path)
    #     try:
    #         x, y, z = img.data.shape
    #         m = min(img.data.shape)
    #         if z!=m:
    #             print("Data is not consistent")               
    #             break
    #     except ValueError:
    #         errors.append((path, img.data.shape))            
            
    # print(errors)
    
    