''' define the transformations to prepare the data for the SCS model
    from the raw patches we obtain from the preprocessing pipeline
'''

import os
import pandas as pd
import torchio as tio
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, SpatialPadd, CenterSpatialCropd, NormalizeIntensityd,
    RandRotated, RandSpatialCropd, RandBiasFieldd, Lambdad, Transform,
    RandGaussianNoised, RandAffined, RandZoomd, Rand3DElasticd, Flipd,
    SpatialCropd, Spacingd, RandLambdad, ResizeWithPadOrCropd,
    RandGaussianSharpend, CenterSpatialCropd, RandScaleIntensityd
)


from torch.utils.data import DataLoader, ConcatDataset
from monai.data import Dataset
import matplotlib.pyplot as plt
import nibabel as nib
from augment import *



# custom transform to extract slices from the 3D image
# and put them in the MIL bag format 
class ExtractSlicesD(Transform):
    def __init__(self, keys=['image'], target_size=(384, 384), verbose=False):
        self.keys = keys
        self.target_size = target_size
        self.resize = tio.Resize(target_shape=(*target_size, 1))
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            # Get image and remove channel dimension (1, X, Y, 6) -> (X, Y, 6)
            image = d[key].squeeze(0)
            for i in range(image.shape[2]):
                # Extract slice, add channel dim for torchio,
                # resize, then normalize
                slice_2d = image[:, :, i]
                slice_3d = slice_2d.unsqueeze(0).unsqueeze(-1)
                if self.verbose:
                    print(f"Shape before resize: {slice_3d.shape}")
                slice_resized = self.resize(slice_3d)
                if self.verbose:
                    print(f"Shape after resize: {slice_resized.shape}")
                # Remove the z dimension that we added
                slice_final = slice_resized.squeeze(-1)
                d[f'slice_{i}'] = slice_final
                if self.verbose:
                    print(f"Final slice {i} shape: {slice_final.shape}")
        return d

# transformation pipeline for the data
def get_transforms_scs(mode='basic'):
    
    regular_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=['image'], pixdim=(0.4, 0.4, 4.4), mode=('bilinear')),  # Ré-échantillonnage de l'image
        SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),  # Padding pour atteindre une taille fixe
        
    ])
    
    
    if mode == 'basic':
        common_transforms = Compose([
            CenterSpatialCropd(keys=['image'],roi_size=(120, 80, 6)),
            ScaleIntensityd(keys=['image']),
            NormalizeIntensityd(keys=['image'],nonzero=True),
        ])

    elif mode == 'random':
        # Same transforms but with random augmentations
        common_transforms = Compose([
            RandRotated(keys=['image'], prob=0.5, range_y=0.1),
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)), 
            RandSpatialCropd(keys=['image'], roi_size=(120, 80, 6), random_size=False),  
            RandLambdad(keys=['image'],func=aug_sqrt,prob=0.05,),
            RandLambdad(keys=['image'],func=aug_sin,prob=0.05,),
            RandLambdad(keys=['image'],func=aug_exp,prob=0.05,),
            RandLambdad(keys=['image'],func=aug_sig,prob=0.05, ),
            RandLambdad(keys=['image'],func=aug_laplace,prob=0.05,),
            RandLambdad(keys=['image'],func=aug_inverse,prob=0.05, ),   
            RandBiasFieldd(keys=['image'],prob=0.05),
            RandAffined(keys=['image'],prob=0.05, padding_mode="zeros", mode=["bilinear"]), 

            RandGaussianNoised(keys=['image'], mean=0.0, std=0.1, prob=0.05),
            RandGaussianSharpend(keys=['image'], prob=0.05),   

            #Rand3DElasticd(keys=['image'],prob=0.05, padding_mode="zeros", mode=["bilinear"], sigma_range=(5,7), magnitude_range=(50,150)),

            ResizeWithPadOrCropd(keys=['image'], spatial_size=(120, 80, 6)),
            RandScaleIntensityd(keys=['image'], factors=(0.8, 1.2), prob=1),   
        ])

    # Create list of transforms for processing 2D slices
    slice_transforms = Compose([
        # Custom transform to extract and resize slices
        ExtractSlicesD(keys=['image'], target_size=(384, 384)),
        
        # Ensure all slices are tensors
        ToTensord(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        # Concatenate all slices into a bag
        ConcatItemsd(
            keys=[f'slice_{i}' for i in range(6)],
            name='bag',
            dim=0
        ),
        # Add a transform to ensure bag has the correct shape
        Lambdad(
            keys=['bag'],
            func=lambda x: x.reshape(6, 1, 384, 384)
        )
    ])

    # Combine common_transforms with slice_transforms
    transforms = Compose([regular_transforms, common_transforms, slice_transforms])

    return transforms

# prepare the data for the SCS model, 
# random is for training, basic for validation
# returns a ConcatDataset of the left and right data
def prepare_data_scs(data_dir, csv_file, random=True):
    data = []
    labels_df = pd.read_csv(csv_file)

    counter = 0
    # Label conversion dictionary
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

    for subject in os.listdir(data_dir):
        print(subject)
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if '_patch.nii.gz' in file and 'foramen' not in file:
                    image_path = os.path.join(subject_dir, file)
                    parts = image_path.split('_')
                    disk_level = f"{parts[-3]}_{parts[-2]}"

                    if os.path.exists(image_path):
                        
                        subject_id = (subject.replace('sub-', ''))

                        label_column = (
                            f'spinal_canal_stenosis_{disk_level.lower()}'
                        )
                        # Get raw label
                        label = labels_df.loc[
                            labels_df['study_id'] == int(subject_id),
                            label_column
                        ].values[0]

                        # Convert text label to numeric value
                        label_numeric = text2int.get(label, -1)
                        if label_numeric != -1:
                            counter += 1
                            data.append({
                                "image": image_path,
                                "label": label_numeric
                            })

    print(f"Number of loaded data: {counter}")
    return Dataset(data=data, transform=get_transforms_scs(mode='random') if random else get_transforms_scs(mode='basic'))