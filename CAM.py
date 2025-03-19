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
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, NormalizeIntensityd, Spacingd, ResizeWithPadOrCropd
)
from monai.networks.nets import ResNet
import torch.optim as optim
from torchcam.methods import SmoothGradCAMpp


# Weight tensor for weighted loss function (if needed)
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

# Transformation pipeline for data
def get_transforms():
    transform = Compose([
        LoadImaged(keys=['T1', 'T2']),
        Spacingd(keys=['T1', 'T2'], pixdim=(2.4, 0.6, 0.6), mode=('bilinear')),
        EnsureChannelFirstd(keys=['T1', 'T2']),
        ResizeWithPadOrCropd(keys=['T1', 'T2'], spatial_size=(10, 70, 70)),
        ScaleIntensityd(keys=['T1', 'T2']),
        NormalizeIntensityd(keys=['T1', 'T2'], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["T1", "T2"], name="combinaison"),
        ToTensord(keys=["combinaison"])
    ])
    return transform

# Prepare data function
def prepare_data(data_dir, csv_file, transform):
    data = []
    labels_df = pd.read_csv(csv_file)
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    counter = 0 
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject, 'anat')
        """if counter > 5: 
            break """
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
                    t1_path = os.path.join(subject_dir, file)
                    parts = t1_path.split('_')
                    disk_level = f"{parts[-5]}_{parts[-4]}"
                    for t2_file in os.listdir(subject_dir):
                        if disk_level in t2_file and 'foramen' in t2_file and 'T2w' in t2_file:
                            t2_path = os.path.join(subject_dir, t2_file)
                    if os.path.exists(t1_path):
                        t1_image = nib.load(t1_path)
                        t2_image = nib.load(t2_path)
                        t1_image_data = t1_image.get_fdata()
                        t2_image_data = t2_image.get_fdata()
                        if t1_image_data.ndim == 3 and t2_image_data.ndim == 3:
                            subject_id = subject.replace('sub-', '')
                            label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}' if 'left' in file else f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                            label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                            label_numeric = text2int.get(label, -1)
                            if label_numeric != -1:
                                data.append({"T1": t1_path, "T2": t2_path, "label": label_numeric, "combinaison": None})
                                counter += 1 
    return Dataset(data=data, transform=transform)

# Inference and confusion matrix function
def inference_and_confusion_matrix(model, val_loader, device):
    model.eval()
    #cam_extractor = SmoothGradCAMpp(model)
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        print(len(val_loader))
        for batch in val_loader:
            inputs = batch["combinaison"].cuda()
            labels = batch["label"].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            print(f'label{labels.cpu().numpy()}')
            print(f'predicted{predicted.cpu().numpy()}')
            """activation_map = cam_extractor(outputs.squeeze(0).argmax().item(), outputs)
            plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
            """
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Moderate", "Severe"], yticklabels=["Normal", "Moderate", "Severe"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_nfn.png')

# Inference only function
def run_inference(device, model_weights_path, val_dir, csv_file, batch_size=4):
    transform = get_transforms()
    val_dataset = prepare_data(val_dir, csv_file, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pre-trained model
    model = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],  # Use the same layers as during training
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=2,
        num_classes=3,
    ).to(device)

    model.load_state_dict(torch.load(model_weights_path))
    
    # Run inference and compute confusion matrix
    inference_and_confusion_matrix(model, val_loader, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weights_path = "model/model_with_loss_56_new_images.pth"  # Specify the model weights path
    val_dir = "../../duke/public/rsna_challenge/20250212nii_data_splits/validation"
    csv_file = "../../duke/public/rsna_challenge/dcom_data/train.csv"
    run_inference(device, model_weights_path, val_dir, csv_file, batch_size=4)

if __name__ == "__main__":
    main()
