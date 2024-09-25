import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
import monai
from monai.transforms import (
    Compose,
    Resize,
    EnsureChannelFirst,
    LoadImage,
    RandRotate,
    Orientation,
)
from monai.data import DataLoader, Dataset
from monai.apps import download_and_extract
import SimpleITK as sitk
import torch


def get_orientation(dicom_file_path):
    # Load the DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Retrieve the Image Orientation (Patient) tag
    orientation = dicom_data.ImageOrientationPatient

    if len(orientation) != 6:
        raise ValueError("Image Orientation (Patient) should have 6 elements.")

    # Convert orientation to numpy array for easier manipulation
    orientation = np.array(orientation).reshape(2, 3)

    # Define the direction mapping
    direction_map = {
        (1, "X"): "R",  # Right
        (-1, "X"): "L",  # Left
        (1, "Y"): "A",  # Anterior
        (-1, "Y"): "P",  # Posterior
        (1, "Z"): "S",  # Superior
        (-1, "Z"): "I",  # Inferior
    }

    # Get the direction cosines for the row and column
    row_cosines = orientation[0, :]
    column_cosines = orientation[1, :]

    # Determine the orientation along each axis
    def determine_direction(cosines, axis):
        if cosines[axis] > 0:
            return direction_map[(1, axis)]
        else:
            return direction_map[(-1, axis)]

    # Determine direction for each axis
    row_direction = determine_direction(row_cosines, 0)  # X-axis
    column_direction = determine_direction(column_cosines, 1)  # Y-axis
    normal_direction = determine_direction(
        np.cross(row_cosines, column_cosines), 2
    )  # Z-axis

    # Combine directions into a single string
    orientation_string = row_direction + column_direction + normal_direction

    return orientation_string


def read_dicom_as_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    image_array = dicom.pixel_array
    spacing = (dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness)
    return image_array, spacing


def save_dicom(image_array, original_dicom_path, output_dicom_path, spacing):
    dicom = pydicom.dcmread(original_dicom_path)
    dicom.PixelData = image_array.tobytes()
    dicom.PixelSpacing = [spacing[0], spacing[1]]
    dicom.save_as(output_dicom_path)


def resample_image(image_array, input_dicom_path, original_spacing, new_spacing):

    # Define transforms
    transform = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            Resize(
                spatial_size=(
                    int(image_array.shape[0] * (original_spacing[0] / new_spacing[0])),
                    int(image_array.shape[1] * (original_spacing[1] / new_spacing[1])),
                )
            ),
            RandRotate(
                range_x=[np.pi / 2, np.pi / 2], range_y=[0, 0], range_z=[0, 0], prob=1
            ),
        ]
    )
    # Apply transform
    resampled_image = transform(input_dicom_path)

    # Convert back to numpy array
    resampled_image = resampled_image.squeeze().numpy()
    return resampled_image


def main():

    description = pd.read_csv("data/train_series_descriptions.csv")
    description = description[description["series_description"] == "Sagittal T2/STIR"]
    study_id, series_id, _ = description.iloc[0]
    print(study_id, series_id)
    description = description.values

    n = len(os.listdir(f"data/train_images/{study_id}/{series_id}"))

    input_dicom_path = f"data/train_images/{study_id}/{series_id}/{str(n//2)}.dcm"
    output_dicom_path = "data/output_resampled.dcm"
    new_spacing = (0.5, 0.5)  # New spacing in mm

    # Read DICOM image
    image_array, original_spacing = read_dicom_as_image(input_dicom_path)
    print(f"Original spacing: {original_spacing}")

    # Resample image
    resampled_image = resample_image(
        image_array, input_dicom_path, original_spacing, new_spacing
    )

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    ax[0].imshow(image_array)
    ax[1].imshow(resampled_image[:, ::-1])
    plt.savefig("resampling_test.png")
    plt.show()

    # Save resampled image
    # save_dicom(resampled_image, input_dicom_path, output_dicom_path, new_spacing)
    # print(f"Resampled DICOM saved to {output_dicom_path}")

    resolutions = {"Sagittal T1": [], "Axial T2": [], "Sagittal T2/STIR": []}

    orientations = {"Sagittal T1": [], "Axial T2": [], "Sagittal T2/STIR": []}

    exclude = []

    for study_id, series_id, seq in description:
        try:
            input_dicom_path = f"data/train_images/{study_id}/{series_id}/1.dcm"
            _, original_spacing = read_dicom_as_image(input_dicom_path)
            orientation = get_orientation(input_dicom_path)
            print(orientation)
            resolutions[seq].append(original_spacing)
            orientations[seq].append(orientation)
        except FileNotFoundError:
            exclude.append(study_id)


if __name__ == "__main__":
    main()
