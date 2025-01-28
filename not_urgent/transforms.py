'File to transforms'
# Importing the necessary libraries
from monai.transforms import (Compose, LoadImaged, Spacingd, EnsureChannelFirstd, SpatialPadd, CenterSpatialCropd, ScaleIntensityd, 
                            NormalizeIntensityd, RandRotated, RandSpatialCropd, ResizeWithPadOrCropd, ToTensord, ConcatItemsd)



# Transforms :
def get_transforms(mode='basic'):
        # Define the transform pipeline with rotation augmentation
    

    if mode == 'basic':
        common_transforms = Compose([
            LoadImaged(keys=['T1','T2']),  # Charge l'image et la segmentation
            Spacingd(keys=['T1','T2'], pixdim=(2,4, 0.6, 0.6), mode=('bilinear')),  # Ré-échantillonnage de l'image
            EnsureChannelFirstd(keys=['T1','T2']),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            SpatialPadd(keys=['T1','T2'], spatial_size=(10, 70, 70)),  # Padding pour atteindre une taille fixe
            CenterSpatialCropd(keys=['T1','T2'], roi_size=(10, 70, 70)),  # Crop pour obtenir une taille fixe
            ScaleIntensityd(keys=['T1','T2']),  # Normalisation de l'intensité pour l'image
            NormalizeIntensityd(keys=['T1','T2'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
            ConcatItemsd(keys=["T1","T2"], name="combinaison"),
            ToTensord(keys=["combinaison"]) 
        ])
    elif mode == 'random':
        # same but changing steps as random steps
        common_transforms = Compose([
            LoadImaged(keys=['T1','T2']),  # Charge l'image et la segmentation
            Spacingd(keys=['T1','T2'], pixdim=(2,4, 0.6, 0.6), mode=('bilinear')),  # Ré-échantillonnage de l'image
            EnsureChannelFirstd(keys=['T1','T2']),  # S'assure que l'image et la segmentation ont la dimension de canal en premier
            RandRotated(keys=['T1','T2'], prob=1, range_y=0.1),  # Rotation aléatoire
            SpatialPadd(keys=['T1','T2'], spatial_size=(10, 70, 70)),  # Padding pour atteindre une taille fixe
            RandSpatialCropd(keys=['T1','T2'], roi_size=(10, 70, 70), random_size=False),  # Crop pour obtenir une taille fixe
            ResizeWithPadOrCropd(keys=['T1', 'T2'], spatial_size=(10, 70, 70)),
            ScaleIntensityd(keys=['T1','T2']),  # Normalisation de l'intensité pour l'image
            NormalizeIntensityd(keys=['T1','T2'], nonzero=True, channel_wise=True),  # Normalisation de l'intensité sur l'image
            ConcatItemsd(keys=["T1","T2"], name="combinaison"),
            ToTensord(keys=["combinaison"]) 
        ])
    return common_transforms
    