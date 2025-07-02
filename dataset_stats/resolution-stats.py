import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import argparse

def calculate_3d_resolutions(csv_file, global_directory, output_dir):
    axial_resolutions = []
    sagittal_t1_resolutions = []
    sagittal_t2_resolutions = []

    with open(csv_file, 'r') as f:
        lines = f.readlines()[1:]

    for line in lines:
        study_id, series_id, series_description = line.strip().split(',')
        series_path = os.path.join(global_directory, study_id, series_id)
        try:
            d1 = pydicom.dcmread(os.path.join(series_path, "1.dcm"))
            d2 = pydicom.dcmread(os.path.join(series_path, "2.dcm"))

            if hasattr(d1, "PixelSpacing"):
                pixel_spacing = [float(v) for v in d1.PixelSpacing]
            else:
                continue

            if hasattr(d1, "ImagePositionPatient") and hasattr(d2, "ImagePositionPatient"):
                pos1 = np.array(d1.ImagePositionPatient, dtype=float)
                pos2 = np.array(d2.ImagePositionPatient, dtype=float)
                slice_thickness = np.linalg.norm(pos2 - pos1)
            else:
                continue

            resolution = pixel_spacing + [slice_thickness]
            if "Axial T2" in series_description:
                axial_resolutions.append(resolution)
            elif "Sagittal T1" in series_description:
                sagittal_t1_resolutions.append(resolution)
            elif "Sagittal T2" in series_description:
                sagittal_t2_resolutions.append(resolution)
        except:
            print(f"Error reading DICOM files for {study_id}, {series_id}. Skipping...")
            continue

    def plot_histograms(resolutions, title, filename):
        if resolutions:
            resolutions = np.array(resolutions)
            fig, axes = plt.subplots(1, 3, figsize=(15,5))
            for i, ax in enumerate(axes):
                ax.hist(resolutions[:, i], bins=20, alpha=0.7, color='blue')
                mean_val = np.mean(resolutions[:, i])
                median_val = np.median(resolutions[:, i])
                std_val = np.std(resolutions[:, i])
                ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
                ax.text(0.95, 0.85, f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}",
                        transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
                ax.set_title(["X-axis", "Y-axis", "Z-axis"][i])
                ax.set_xlabel("Resolution (mm)")
                ax.set_ylabel("Frequency")
                ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

    os.makedirs(output_dir, exist_ok=True)
    plot_histograms(axial_resolutions, "Axial T2", "axial_t2_resolutions.jpg")
    plot_histograms(sagittal_t1_resolutions, "Sagittal T1", "sagittal_t1_resolutions.jpg")
    plot_histograms(sagittal_t2_resolutions, "Sagittal T2", "sagittal_t2_resolutions.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    calculate_3d_resolutions(args.csv_path, args.dicom_dir, args.output)
