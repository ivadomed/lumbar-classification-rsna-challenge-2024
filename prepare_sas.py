''' define the transformations to prepare the data for the SAS model
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
    SpatialCropd, Spacingd
)

from torch.utils.data import DataLoader, ConcatDataset
from monai.data import Dataset
import matplotlib.pyplot as plt
import nibabel as nib


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

# whole transforms for dataloading 
def get_transforms_sas(mode='basic', side = "right"):
    # Define the transform pipeline with rotation augmentation
    
    regular_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
    ])

    if side == "left": # for side (flip or not wether left or right)
        regular_transforms = Compose([regular_transforms,Flipd(keys=['image'], spatial_axis=0)])


    if mode == 'basic':
        common_transforms = Compose([
            SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 6)),  # crop pour récupérer la gauche
            SpatialPadd(keys=['image'], spatial_size=(60, 80, 6)),  # Padding pour atteindre une taille fixe
            CenterSpatialCropd(keys=['image'], roi_size=(60, 80, 6))  # Crop pour obtenir une taille fixe
        ])
    
    elif mode == 'random': # for training !
        common_transforms = Compose([
            RandRotated(keys=['image'], prob=1, range_x=0.2),
            SpatialCropd(keys=['image'], roi_start=(0, 0, 0), roi_end=(80, 100, 6)),  # crop pour récupérer la gauche
            SpatialPadd(keys=['image'], spatial_size=(60, 80, 6)),  # Padding pour atteindre une taille fixe
            RandAffined(keys=['image'], prob=1.0, shear_range=(0.3, 0.3, 0.3)),
            Rand3DElasticd(keys=['image'], prob=0.5, sigma_range=(8, 12), magnitude_range=(100, 200)),
            RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.1),
            RandBiasFieldd(keys=['image'], prob=0.5, coeff_range=(0, 0.4)),
            RandZoomd(keys=['image'], prob=0.5, min_zoom=0.95, max_zoom=1.15),
            RandSpatialCropd(keys=['image'], roi_size=(60, 80, 6), random_size=False)  # Crop pour obtenir une taille fixe
        ])

    # Create list of transforms for processing 2D slices
    slice_transforms = Compose([
        # Custom transform to extract and resize slices
        ExtractSlicesD(keys=['image'], target_size=(384, 384)),
        # Scale and normalize
        ScaleIntensityd(
            keys=[f'slice_{i}' for i in range(6)]
        ),
        NormalizeIntensityd(
            keys=[f'slice_{i}' for i in range(6)],
            nonzero=True
        ),
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

# prepare the data for the SAS model, 
# random is for training, basic for validation
# returns a ConcatDataset of the left and right data
def prepare_data_sas(data_dir, csv_file, random=True):
    data_left = []
    data_right = []
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

                        label_column_sasl = f'left_subarticular_stenosis_{disk_level.lower()}'
                        label_column_sasr = f'right_subarticular_stenosis_{disk_level.lower()}'
                        # Obtenir l'étiquette brute

                        label_sasr = labels_df.loc[labels_df['study_id'] == subject_id, label_column_sasl].values[0]
                        label_sasl = labels_df.loc[labels_df['study_id'] == subject_id, label_column_sasr].values[0]
                        
                        # Convertir l'étiquette textuelle en valeur numérique
                        label_numeric_sasr = text2int.get(label_sasr, -1)
                        label_numeric_sasl = text2int.get(label_sasl, -1)
                        if label_numeric_sasr in [0, 1, 2] and label_numeric_sasl in [0, 1, 2]:
                            data_right.append({"image": image_path, "label": label_numeric_sasr})
                            data_left.append({"image": image_path, "label": label_numeric_sasl})
                            counter += 2
 
    print(f"Number of loaded data: {counter}")
    return Dataset(data=data_left, transform= get_transforms_sas(mode='random', side='left') if random else get_transforms_sas(mode='basic', side='left')), Dataset(data=data_right, transform= get_transforms_sas(mode='random', side='right') if random else get_transforms_sas(mode='basic', side='right') )
