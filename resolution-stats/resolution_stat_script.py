# this scripts aims to plot histograms, mean and standard deviation of the resolutions of the 3D volumes of the MRI images
# this is done using the csv file "train_series_descriptions.csv" that contains the series descriptions of the MRI images
# and displays 3 different plots for the axial T2, sagittal T1 and sagittal T2 resolutions


import os
import pandas
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def calculate_3d_resolutions(csv_file, global_directory):
    # Listes pour les résolutions des trois types d'acquisitions
    axial_resolutions = []
    sagittal_t1_resolutions = []
    sagittal_t2_resolutions = []

    # Lire le fichier CSV
    with open(csv_file, 'r') as f:
        lines = f.readlines()[1:]  # Ignorer l'en-tête

    # Parcourir chaque ligne
    for line in lines:
        study_id, series_id, series_description = line.strip().split(',')

        # Chemin vers la série
        series_path = os.path.join(global_directory, study_id, series_id)

        try:
            # Lire les fichiers 1.dcm et 2.dcm
            dicom_file_1 = pydicom.dcmread(os.path.join(series_path, "1.dcm"))
            dicom_file_2 = pydicom.dcmread(os.path.join(series_path, "2.dcm"))

            # Résolution 2D par slice
            if hasattr(dicom_file_1, "PixelSpacing"):
                pixel_spacing = [float(v) for v in dicom_file_1.PixelSpacing]
            else:
                print(f"PixelSpacing not found for {series_path}")
                continue

            # Résolution 3D (espacement entre les plans)
            if hasattr(dicom_file_1, "ImagePositionPatient") and hasattr(dicom_file_2, "ImagePositionPatient"):
                position_1 = np.array(dicom_file_1.ImagePositionPatient, dtype=float)
                position_2 = np.array(dicom_file_2.ImagePositionPatient, dtype=float)
                slice_thickness = np.linalg.norm(position_2 - position_1)
            else:
                print(f"ImagePositionPatient not found for {series_path}")
                continue

            # Ajouter les résolutions à la liste appropriée
            resolution = pixel_spacing + [slice_thickness]
            if "Axial T2" in series_description:
                axial_resolutions.append(resolution)
            elif "Sagittal T1" in series_description:
                sagittal_t1_resolutions.append(resolution)
            elif "Sagittal T2" in series_description:
                sagittal_t2_resolutions.append(resolution)
        except Exception as e:
            print(f"Error reading DICOM files in {series_path}: {e}")

    # Créer des histogrammes pour chaque type d'acquisition
    def plot_histograms(resolutions, title, save_path):
        if resolutions:
            resolutions = np.array(resolutions)
            axes = ['X-axis', 'Y-axis', 'Z-axis']
            fig, axes_array = plt.subplots(1, 3, figsize=(15, 5))

            for i, ax in enumerate(axes_array):
                ax.hist(resolutions[:, i], bins=20, alpha=0.7, color='blue')
                mean_val = np.mean(resolutions[:, i])
                std_val = np.std(resolutions[:, i])
                ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
                ax.text(0.95, 0.85, f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
                        transform=ax.transAxes, fontsize=10, ha='right', va='top', color='black',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
                ax.set_title(f"{axes[i]} {title}")
                ax.set_xlabel("Resolution (mm)")
                ax.set_ylabel("Frequency")
                ax.legend()

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

    plot_histograms(axial_resolutions, "Axial T2 Resolutions", "axial_t2_resolutions.jpg")
    plot_histograms(sagittal_t1_resolutions, "Sagittal T1 Resolutions", "sagittal_t1_resolutions.jpg")
    plot_histograms(sagittal_t2_resolutions, "Sagittal T2 Resolutions", "sagittal_t2_resolutions.jpg")
    print("Histograms saved.")

# Exécution du script
'''csv_path = "train_series_descriptions.csv"
dicom_base_dir = "train_images"
output_plot_path = "stats"
calculate_3d_resolutions(csv_path, dicom_base_dir)
'''