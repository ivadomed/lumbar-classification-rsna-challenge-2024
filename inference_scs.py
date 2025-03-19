import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    ToTensord, NormalizeIntensityd, Spacingd, SpatialPadd, CenterSpatialCropd
)
from monai.networks.nets import ResNet

def get_transforms():
    transform = Compose([
        LoadImaged(keys=['image']),
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),
        EnsureChannelFirstd(keys=['image']),
        SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
        CenterSpatialCropd(keys=['image'], roi_size=(120, 80, 6)),
        ScaleIntensityd(keys=['image']),
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        ToTensord(keys=['image'])
    ])
    return transform

def prepare_data(data_dir, csv_file, transform):
    data = []
    labels_df = pd.read_csv(csv_file)
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"
                    if os.path.exists(image_path):
                        image_data = nib.load(image_path).get_fdata()
                        if image_data.ndim == 3:
                            subject_id = subject.replace('sub-', '')
                            label_column = f'spinal_canal_stenosis_{disk_level.lower()}'
                            label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                            label_numeric = text2int.get(label, -1)
                            if label_numeric != -1:
                                data.append({"image": image_path, "label": label_numeric})
    return Dataset(data=data, transform=transform)

def inference_and_confusion_matrix(model, val_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].cuda()
            labels = batch["label"].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Moderate", "Severe"], yticklabels=["Normal", "Moderate", "Severe"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_scs.png')

def run_inference(device, model_weights_path, val_dir, csv_file, batch_size=4):
    transform = get_transforms()
    val_dataset = prepare_data(val_dir, csv_file, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)
    model.load_state_dict(torch.load(model_weights_path))
    inference_and_confusion_matrix(model, val_loader, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weights_path = "scs_model_layers_[3, 4, 6, 3]_epochs_40_lr_0.0001_augmentation_True_wd_0.0001_3times.pth"
    val_dir = "../../duke/public/rsna_challenge/20250102nii_data_splits/validation"
    csv_file = "../../duke/public/rsna_challenge/dcom_data/train.csv"
    run_inference(device, model_weights_path, val_dir, csv_file, batch_size=4)

if __name__ == "__main__":
    main()
