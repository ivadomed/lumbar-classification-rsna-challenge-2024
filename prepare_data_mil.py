import os
import pandas as pd
import torchio as tio
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ConcatItemsd,
    ToTensord, SpatialPadd, CenterSpatialCropd, NormalizeIntensityd,
    RandRotated, RandSpatialCropd, RandBiasFieldd, Lambdad, Transform,
    RandGaussianNoised, RandAffined, RandZoomd, Rand3DElasticd
)
from torch.utils.data import DataLoader
from monai.data import Dataset
import matplotlib.pyplot as plt
import nibabel as nib


# We use patches for SCS extracted with these values in mm:
# 'RL': 60,  Right-Left
# 'AP': 40, Anterior-Posterior
# 'SI': 30, Superior-Inferior

# The median value for the voxel size is 0.43 mm in the axial plane
# and 4.4 mm between axial planes (slice thickness)
# So basically we're having a patch of size 150*100*7 for 0.4,0.4,4.4
# resampling

# Extract slices from the image and resize them to the target size
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

# regular transforms just creating the dataset
regular_transforms = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=["image"]),
    ])

# get transforms to call at each getitem
def get_transforms_sas(mode='basic', side='left'):
    if mode == 'basic':
        common_transforms = Compose([
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
            CenterSpatialCropd(
                keys=['image'],
                roi_size=(120, 80, 6)
            ),
        ])

    elif mode == 'random':
        # Same transforms but with random augmentations
        common_transforms = Compose([
            RandRotated(keys=['image'], prob=0.8, range_x=0.2),
            RandGaussianNoised(keys=['image'], prob=0.4, mean=0.0, std=0.1),
            RandBiasFieldd(keys=['image'], prob=0.4, coeff_range=(0, 0.3)),
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
            RandSpatialCropd(
                keys=['image'],
                roi_size=(120, 80, 6),
                random_size=False
            ),
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
    transforms = Compose([common_transforms, slice_transforms])

    return transforms

# get transforms to call at each getitem
def get_transforms(mode='basic'):
    if mode == 'basic':
        common_transforms = Compose([
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
            CenterSpatialCropd(
                keys=['image'],
                roi_size=(120, 80, 6)
            ),
        ])

    elif mode == 'random':
        # Same transforms but with random augmentations
        common_transforms = Compose([
            RandRotated(keys=['image'], prob=1.0, range_x=0.2),
            RandAffined(keys=['image'], prob=1.0, shear_range=(0.3, 0.3, 0.3)),
            Rand3DElasticd(keys=['image'], prob=0.5, sigma_range=(8, 12), magnitude_range=(100, 200)),
            RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.1),
            RandBiasFieldd(keys=['image'], prob=0.5, coeff_range=(0, 0.4)),
            RandZoomd(keys=['image'], prob=0.5, min_zoom=0.95, max_zoom=1.15),
            SpatialPadd(keys=['image'], spatial_size=(120, 80, 6)),
            RandSpatialCropd(
                keys=['image'],
                roi_size=(120, 80, 6),
                random_size=False
            ),
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
    transforms = Compose([common_transforms, slice_transforms])

    return transforms

# custom dataset to apply transforms at each getitem
class CustomDataset(Dataset):
    def __init__(self, data, fixed_transform=regular_transforms, random_transform=True):
        self.data = [fixed_transform(d) for d in data] if fixed_transform else data 
        self.random_transform = random_transform

    def __getitem__(self, index):
        data = self.data[index]
        if self.random_transform:
            data = get_transforms(mode='random')(data)
        else :
            data = get_transforms(mode='basic')(data)
        return data


def prepare_data(data_dir, csv_file, random=True):
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
                        # Check image shape
                        image_data = nib.load(image_path).get_fdata()
                        if image_data.ndim == 3:
                            subject_id = (subject.replace('sub-', ''))

                            label_column = (
                                f'spinal_canal_stenosis_{disk_level.lower()}'
                            )
                            # Get raw label
                            label = labels_df.loc[
                                labels_df['study_id'] == subject_id,
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
    return CustomDataset(data=data, fixed_transform=regular_transforms, random_transform=random)


# function to test the data preparation
# load using torch.utils.data.DataLoader
# then print the shape of the data
# also for the first batch, print the label
# and plot the slices
def test_data_preparation(data_dir, csv_file, transform, batches=1):
    data = prepare_data(data_dir, csv_file, transform)
    dataloader = DataLoader(data, batch_size=8, shuffle=False)
    i = 0
    for batch in dataloader:
        # Add debug print to see all available keys
        if i < batches:
            print("Available keys in batch:", batch.keys())
            if 'bag' in batch:
                print("Bag shape:", batch['bag'].shape)
                print("Label:", batch['label'])
                for bag_idx, bag in enumerate(batch['bag']):
                    # display the bag as a 3x2 grid, gray scale
                    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                    # Move suptitle to after the loop to avoid duplicate titles
                    for slice_idx in range(6):
                        row, col = slice_idx // 2, slice_idx % 2
                        # Get the correct slice and resize it
                        slice_img = bag[slice_idx].squeeze(0).numpy()
                        # Shape devrait être (384, 384)
                        axs[row, col].imshow(slice_img, cmap='gray')
                        axs[row, col].set_title(f'Slice {slice_idx}')
                        axs[row, col].axis('off')
                        # Add a title to the figure
                        # and the label
                        fig.suptitle(
                            f'Sample {bag_idx}, '
                            f'Label: {batch["label"][bag_idx].item()}'
                        )
                    plt.tight_layout()
                    plt.savefig(f'sample_1_{bag_idx}.png')
            i += 1


# for testing
'''data_dir = ('C:/Users/abels/OneDrive/Documents/NeuroPoly/rsna-challenge/sample_test')
csv_file = ('C:/Users/abels/OneDrive/Documents/NeuroPoly/rsna-challenge/sample_test/train.csv')
transform = get_transforms(mode='random')
test_data_preparation(data_dir, csv_file, transform)
'''