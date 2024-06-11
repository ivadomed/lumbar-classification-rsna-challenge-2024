import numpy as np
import pandas as pd
import os
import subprocess
import nibabel as nib
import pydicom
from sklearn.model_selection import train_test_split
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
import subprocess

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# Define a custom dataset class
class Dataset_2D(Dataset):
    def __init__(self, data_df, transform=None, infer=False):
        labels = data_df.drop(columns=["img_path","study_id"])
        # convert labels to a list of intergers lists
        labels = labels.values.tolist()

        self.data = {"paths": data_df["img_path"], "labels": labels, "study_id": data_df["study_id"]}
        self.transform = transform
        self.length = len(self.data["paths"])
        self.num_classes = len(labels[0])
        self.infer = infer
    def __len__(self):
        return len(self.data["paths"])

    def __getitem__(self, index):
        print(index)
        path = self.data["paths"][index]
        if not self.infer:
            label = self.data["labels"][index]
            # convert label list to tensor with shape [1,num_classes]
            label = torch.tensor([label])
        print(path)
        if self.transform:
            ## Rotate, flip, shift intensity if training
            image = self.transform(path)

        
        print(image.shape)
        if image.shape[0] > 1:
            # Some images have multiple channels, average the channels
            image = torch.mean(image, dim=0).unsqueeze(0)

        #add channel dimension
        image = image.unsqueeze(0)

        if not self.infer:
            return image, label
        else:
            return image, self.data["study_id"][index]

def load_dcm(root_folder):
    """ Load the DICOM files from the specified folder and return the 3D volume."""
    slices = [pydicom.dcmread(os.path.join(root_folder, s)) for s in os.listdir(root_folder)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    volume = np.stack([s.pixel_array for s in slices])
    return volume

def verbal_to_vector(labels_df_line):
    # convert the panda line to a list of values
    labels_df_line = labels_df_line.values.tolist()[0]
    label = np.zeros(75)
    for i in range(0,25):
        if labels_df_line[i+1] == "Normal/Mild":
            label[3*i+0]=1
        elif labels_df_line[i+1] == "Moderate":
            label[3*i+1]=1
        elif labels_df_line[i+1] == "Severe":
            label[3*i+2]=1
    return label

def build_data(labels_df, root_dir):
    """ find the path of the nifti files in the root_dir and link them to the labels_df."""
    # go through the root_dir and find the path of the nii.gz files is os
    labels = []
    labels_df = pd.read_csv(labels_df)
    for deasease in labels_df.columns[1:].tolist():
        labels.append(deasease + "_Normal/Mild")
        labels.append(deasease + "_Moderate")
        labels.append(deasease + "_Severe")

    data_df = pd.DataFrame(columns=["img_path", "study_id"] + labels)
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nii.gz"):            
                # get the folder name
                # get the patient ID
                study_id = root.split("/")[-2]
                # Find the line in labels_df that corresponds to the patient ID
                patient_label = verbal_to_vector(labels_df[labels_df["study_id"] == int(study_id)])
                # convert the label to a dataframe with multiple columns
                label_df = pd.DataFrame(patient_label.reshape(1, -1), columns=labels)
                # concatenate the dataframes
                data_df = pd.concat([data_df, pd.DataFrame({"img_path": root+"/"+file, "study_id": int(study_id)}, index=[0]).join(label_df)], ignore_index=True)
    # split the data into train and val
    train_data_df, val_data_df = train_test_split(data_df, test_size=0.2)
    # Reset the index
    train_data_df.reset_index(drop=True, inplace=True)
    val_data_df.reset_index(drop=True, inplace=True)
    # save csv
    train_data_df.to_csv("train_data_nifti.csv")

    return train_data_df, val_data_df, len(labels)

def build_test_data(root_dir):
    """ find the path in the root dir and return them in a the dataframe."""
    data_df = pd.DataFrame(columns=["img_path", "study_id"])
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nii.gz"):            
                # get the folder name
                # get the patient ID
                study_id = root.split("/")[-2]
                # concatenate the dataframes
                data_df = pd.concat([data_df, pd.DataFrame({"img_path": root+"/"+file, "study_id": int(study_id)}, index=[0])], ignore_index=True)
    # Reset the index
    data_df.reset_index(drop=True, inplace=True)
    return data_df

    

def df_to_Dataset(data_df, val = False, infer = False):
    """ Convert the file paths to a custom dataset object."""
    if not val:
        transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
        RandRotate(range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5),
        # limit the size of the image to 640x640x64 to avoid memory errors
        CenterSpatialCrop((1024, 1024, 64)),
    ]
)
    else:
        transform = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
    ]
)
    dataset = Dataset_2D(data_df, transform=transform, infer=infer)
    return dataset