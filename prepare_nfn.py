''' define the transformations to prepare the data for the NFN model
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
class ExtractSlicesD_nfn(Transform):
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
            for i in range(image.shape[0]):
                # Extract slice, add channel dim for torchio,
                # resize, then normalize
                slice_2d = image[i, :, :]
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
def get_transforms_nfn(mode='basic', side = "right"):

    regular_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
    ])

    if side == "left": 
        regular_transforms = Compose([regular_transforms,Flipd(keys=['image'], spatial_axis=0)])


    if mode == 'basic':
        common_transforms = Compose([
            #Spacingd(keys=['image'], pixdim=(4.0, 0.4, 0.4), mode=('bilinear')),
            SpatialPadd(keys=['image'], spatial_size=(6, 100, 100)),
            CenterSpatialCropd(
                keys=['image'],
                roi_size=(6, 100, 100)
            ),
        ])

    elif mode == 'random': # for training !
        # Same transforms but with random augmentations
        common_transforms = Compose([
            #Spacingd(keys=['image'], pixdim=(4.0, 0.4, 0.4), mode=('bilinear')),
            RandRotated(keys=['image'], prob=0.8, range_y=0.2),
            RandGaussianNoised(keys=['image'], prob=0.4, mean=0.0, std=0.1),
            RandBiasFieldd(keys=['image'], prob=0.4, coeff_range=(0, 0.3)),
            SpatialPadd(keys=['image'], spatial_size=(6, 100, 100)),
            RandSpatialCropd(
                keys=['image'],
                roi_size=(6, 100, 100),
                random_size=False
            ),
        ])

    # Create list of transforms for processing 2D slices
    slice_transforms = Compose([
        # Custom transform to extract and resize slices
        ExtractSlicesD_nfn(keys=['image'], target_size=(224, 224)),
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
            func=lambda x: x.reshape(6, 1, 224, 224)
        )
    ])

    # Combine common_transforms with slice_transforms
    transforms = Compose([regular_transforms, common_transforms, slice_transforms])

    return transforms

# prepare the data for the NFN model, 
# random is for training, basic for validation
# returns a ConcatDataset of the left and right data
def prepare_data_nfn(data_dir, csv_file, random=True):
    data_right = []
    data_left = []
    labels_df = pd.read_csv(csv_file)

    counter = 0
    # Label conversion dictionary
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

    for subject in os.listdir(data_dir):
        #print(subject)
        """if counter >40: 
            break """
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):
            for file in os.listdir(subject_dir):
                if '_patch.nii.gz' in file and 'foramen' in file and 'T1' in file:
                    image_path = os.path.join(subject_dir, file)
                    parts = image_path.split('_')
                    disk_level = f"{parts[-5]}_{parts[-4]}"

                    if os.path.exists(image_path):
                        
                        subject_id = (subject.replace('sub-', ''))
                        if 'left' in file:
                            orientation = 'right'
                        elif 'right' in file: 
                            orientation = 'left'
                        label_column = (
                            f'{orientation}_neural_foraminal_narrowing_{disk_level.lower()}'
                        )
                        print(file)
                        print(label_column )
                        # Get raw label
                        label = labels_df.loc[
                            labels_df['study_id'] == subject_id,
                            label_column
                        ].values[0]

                        # Convert text label to numeric value
                        label_numeric = text2int.get(label, -1)
                        if label_numeric != -1:
                            counter += 1
                            if "left" in image_path: 
                                data_right.append({
                                    "image": image_path,
                                    "label": label_numeric
                                })
                            if "right" in image_path: 
                                data_left.append({
                                    "image": image_path,
                                    "label": label_numeric
                                })

    print(f"Number of loaded data: {counter}")
    return ConcatDataset([Dataset(data=data_left, transform=get_transforms_nfn(mode='random', side='left') if random else get_transforms_nfn(mode='basic',side='left')), Dataset(data=data_right, transform=get_transforms_nfn(mode='random', side='right') if random else get_transforms_nfn(mode='basic',side='right'))]) 

