import os
import torch
from tqdm import tqdm
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    ToTensord, SpatialPadd, CenterSpatialCropd, Spacingd,
    NormalizeIntensityd, SpatialCropd
)
from monai.networks.nets import ResNet
import argparse 
from torch.utils.data import DataLoader, ConcatDataset
import csv 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_csv', required=True)
    return parser.parse_args()


def get_transforms_sas(left=True):
    
    first_transforms = [
        LoadImaged(keys=['T2']),
        EnsureChannelFirstd(keys=['T2']),
        Spacingd(keys=['T2'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  
        SpatialPadd(keys=['T2'], spatial_size=(120, 80, 6)), 
        CenterSpatialCropd(keys=['T2'], roi_size=(120,80, 6)),  
    ]

    if left: 
        crop_transform = [
            SpatialCropd(keys=["T2"], roi_size=(60, 80, 6), roi_center=(30, 40, 3))
        ]

    else: 
        crop_transform = [
            SpatialCropd(keys=["T2"], roi_size=(60, 80, 6), roi_center=(90, 40, 3))
        ]

    second_transforms_basic = [
        ScaleIntensityd(keys=['T2']), 
        NormalizeIntensityd(keys=['T2'], nonzero=True, channel_wise=True),
        ToTensord(keys=['T2'])
        ]
    
    common_transforms = Compose(first_transforms + crop_transform + second_transforms_basic)
    
    return common_transforms




def prepare_data_sas(data_dir, transform_right, transform_left):
    data_right = []
    data_left = []
    counter = 0 
    
    for subject in os.listdir(data_dir):        

        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                
                if '_patch.nii.gz' in file and 'sag' not in file and 'T2w' in file:
                    t2_path = os.path.join(subject_dir, file)
                    
                    parts = t2_path.split('_')

                    disk_level = f"{parts[-3]}_{parts[-2]}"       

                    if os.path.exists(t2_path):
                        
                        subject_id = (subject.replace('sub-', ''))
                        
                        label_column_left = f'{subject}_left_subarticular_stenosis_{disk_level.lower()}'
                            
                        label_column_right = f'{subject}_right_subarticular_stenosis_{disk_level.lower()}'
                                                          
                        data_left.append({"T2": t2_path, "label": label_column_left})
                        data_right.append({"T2": t2_path, "label": label_column_right})
                        counter +=1

    return ConcatDataset([Dataset(data_left, transform_left),Dataset(data_right, transform_right)])


def inference_sas(device, data_dir, model_path, batch_size=4, layers=[3, 4, 6, 3]):

    transform_right = get_transforms_sas(left = False)
    transform_left = get_transforms_sas(left = True )
    dataset = prepare_data_sas(data_dir, transform_right, transform_left)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    
    """model1 = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model1.load_state_dict(torch.load(f'{model_path}/sas_1.pth', map_location=device))
    model1.eval()"""

    model2 = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model2.load_state_dict(torch.load(f'{model_path}/sas_2.pth', map_location=device))
    model2.eval()

    model3 = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model3.load_state_dict(torch.load(f'{model_path}/sas_3.pth', map_location=device))
    model3.eval()

    model4 = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model4.load_state_dict(torch.load(f'{model_path}/sas_4.pth', map_location=device))
    model4.eval()

    """model5 = ResNet(
        block="bottleneck",
        layers=layers,
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
    ).to(device)

    model5.load_state_dict(torch.load(f'{model_path}/sas_5.pth', map_location=device))
    model5.eval()
    """

    pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch["T2"].to(device)
            labels = batch["label"]

            #outputs1 = model1(inputs) 
            outputs2 = model2(inputs) 
            outputs3 = model3(inputs) 
            outputs4 = model4(inputs) 
            #outputs5 = model5(inputs) 

            outputs = (outputs4 + outputs2 + outputs3)/3 #+ outputs4 + outputs5)/5
            
            outputs = list(outputs.softmax(dim = 1).cpu().numpy())
            
            for i in range(len(labels)): 
                label = labels [i]
                output = list(outputs[i])
                
                pred.append((label, output))
    
    pred_sorted = sorted(pred, key=lambda x: x[0])

    return pred_sorted 

def main():
    args = parse_args()
    data_dir = args.data
    model_path = args.model_path  
    output_csv = args.output_csv


    if not os.path.exists(model_path):
        print(f"Error: Model folder not found at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_sas = inference_sas(
        device=device,
        data_dir=data_dir,
        model_path=model_path,
        batch_size=2,
        layers=[3, 4, 6, 3]
    )


    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["subject", "pathology", "level", "Normal/Mild", "Moderate", "Severe"])
        
        # Write each prediction
        for label, output in pred_sas:
            parts = label.split("_")
            subject = parts[0]  # e.g. sub-121
            pathology = "_".join(parts[1:-2])  # e.g. left_neural_foraminal_narrowing
            level = "_".join(parts[-2:])  # e.g. l5_s1
            
            writer.writerow([subject, pathology, level, round(output[0], 2), round(output[1], 2), round(output[2], 2)])



if __name__ == "__main__":
    main()