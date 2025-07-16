import os
import argparse
import numpy as np
import pandas as pd
import pydicom

def get_axis_labels(row_cos, col_cos, normal):
    """
    Determine the anatomical directions of image axes.
    """
    directions = []
    for vec in [row_cos, col_cos, normal]:
        axis = np.argmax(np.abs(vec))
        sign = vec[axis]
        if axis == 0:
            directions.append('L' if sign >= 0 else 'R')
        elif axis == 1:
            directions.append('P' if sign >= 0 else 'A')
        else:
            directions.append('S' if sign >= 0 else 'I')
    return directions

def determine_flips(dcm):
    iop = np.array(dcm.ImageOrientationPatient)
    row_cos = iop[:3]
    col_cos = iop[3:]
    normal = np.cross(row_cos, col_cos)

    labels = get_axis_labels(row_cos, col_cos, normal)
    desired = ['P', 'I', 'L']  # x, y, z

    flips = []
    for label, want in zip(labels, desired):
        flips.append(label != want)
    return flips

def load_dicom_metadata(dicom_dir, instance_number):
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        try:
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            if hasattr(dcm, "InstanceNumber") and dcm.InstanceNumber == instance_number:
                return dcm
        except:
            continue
    raise ValueError(f"Instance {instance_number} not found in {dicom_dir}")

def reorient_pixel_coords(x, y, z, shape, flips):
    if not flips[0]:  # x: posterior to anterior
        x = shape[1] - 1 - x
    if not flips[1]:  # y: inferior to superior
        y = shape[0] - 1 - y
    if not flips[2]:  # z: left to right
        z = shape[2] - 1 - z
    return x, y, z

def main(args):
    df = pd.read_csv(args.csv_input)

    # Only keep rows with relevant conditions
    df_filtered = df[df["condition"].str.contains("Neural Foraminal Narrowing|Subarticular Stenosis", case=False, na=False)].copy()

    output_rows = []

    for _, row in df_filtered.iterrows():
        study_id = str(row["study_id"])
        series_id = str(row["series_id"])
        instance_number = int(row["instance_number"])
        x_pix = float(row["x"])
        y_pix = float(row["y"])

        dicom_dir = os.path.join(args.dicom_root, study_id, series_id)

        # Load and sort all DICOMs in the series
        dicoms = []
        for fname in os.listdir(dicom_dir):
            fpath = os.path.join(dicom_dir, fname)
            try:
                dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
                if hasattr(dcm, "InstanceNumber"):
                    dicoms.append((dcm.InstanceNumber, dcm))
            except:
                continue

        if not dicoms:
            print(f"Warning: No valid DICOMs found in {dicom_dir}")
            continue

        dicoms.sort(key=lambda x: x[0])
        instance_numbers = [inst for inst, _ in dicoms]
        total_slices = len(instance_numbers)

        try:
            z_index = instance_numbers.index(instance_number)
        except ValueError:
            print(f"Warning: Instance {instance_number} not found in {dicom_dir}")
            continue

        dcm = dicoms[z_index][1]
        shape = (int(dcm.Rows), int(dcm.Columns), total_slices)
        flips = determine_flips(dcm)

        x_new, y_new, z_new = reorient_pixel_coords(x_pix, y_pix, z_index, shape, flips)

        output_rows.append({
            "study_id": study_id,
            "series_id": series_id,
            "instance_number": instance_number,
            "condition": row["condition"],
            "level": row["level"],
            "x_P2A": x_new,
            "y_I2S": y_new,
            "z_L2R": z_new
        })

    # Save as new DataFrame
    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(args.csv_output, index=False)
    print(f"✅ Reoriented CSV saved to: {args.csv_output} (n = {len(df_out)})")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorient DICOM pixel coordinates to canonical frame (P→A, I→S, L→R)")
    parser.add_argument("--dicom_root", type=str, required=True, help="Root directory containing DICOMs organized as study_id/series_id/*.dcm")
    parser.add_argument("--csv_input", type=str, required=True, help="Input CSV with pixel coordinates and instance numbers")
    parser.add_argument("--csv_output", type=str, required=True, help="Output CSV with reoriented coordinates")

    args = parser.parse_args()
    main(args)
