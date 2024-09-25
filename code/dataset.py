import pandas as pd
from tqdm import tqdm
from image import Image
import glob
import numpy as np
import torch
from utils import *
from monai.data import Dataset


class SpinalCanalStenosisDataset(Dataset):
    def __init__(
        self,
        root_dir: str = None,
        vol_paths: list = None,
        seg_paths: list = None,
        labels_csv: str = "./data/train.csv",
        transform: any = None,
        exclude: list = None,
    ):

        text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
        vol_paths.sort()
        seg_paths.sort()
        self.root_dir = root_dir
        self.vol_paths = vol_paths
        self.seg_paths = seg_paths
        self.transform = transform

        self.labels = pd.read_csv(labels_csv)
        self.labels = self.labels[
            [
                "study_id",
                "spinal_canal_stenosis_l1_l2",
                "spinal_canal_stenosis_l2_l3",
                "spinal_canal_stenosis_l3_l4",
                "spinal_canal_stenosis_l4_l5",
                "spinal_canal_stenosis_l5_s1",
            ]
        ]

        self.labels = self.labels.replace(text2int)

        exclude_vol = []
        exclude_seg = []

        if exclude is not None:
            for study_id in exclude:
                for i in range(len(vol_paths)):
                    if "sub-" + str(study_id) in vol_paths[i]:
                        # print(study_id)
                        exclude_vol.append(vol_paths[i])
                        exclude_seg.append(seg_paths[i])

        for x in exclude_vol:
            vol_paths.remove(x)

        for x in exclude_seg:
            seg_paths.remove(x)

    def __len__(self):
        return len(self.vol_paths)

    def __getitem__(self, idx):

        vol_path = self.vol_paths[idx]
        x = vol_path.split("/")[-1]
        x = x[:-7] + "_0000.nii.gz"

        study_id = x.split("_")[0][4:]

        seg_path = self.seg_paths[idx]
        label = (
            self.labels[self.labels["study_id"] == int(study_id)]
            .values[0, 1:]
            .astype(int)
        )
        if label.min() < 0 or label.max() > 2:
            print(study_id)

        vol = Image(self.root_dir + "/output/input/" + x)
        vol.change_orientation("LSA")
        vol = vol.data
        seg = Image(seg_path)
        seg.change_orientation("LSA")
        seg = seg.data

        D, H, W = vol.shape
        discs = np.isin(seg, [202, 203, 204, 205, 206]).astype(int)
        disc_l5 = np.isin(seg, [202]).astype(int)
        disc_l4 = np.isin(seg, [203]).astype(int)
        disc_l3 = np.isin(seg, [204]).astype(int)
        disc_l2 = np.isin(seg, [205]).astype(int)
        disc_l1 = np.isin(seg, [206]).astype(int)
        spinal_canal = np.isin(seg, [201]).astype(int)

        discs = [disc_l1, disc_l2, disc_l3, disc_l4, disc_l5]
        levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        patches = {}
        patches_seg = {}

        w = 20
        for i, disc in enumerate(discs):
            patch = patch_extraction(vol, disc, d=0, h=20, w=20)
            patches[levels[i]] = torch.Tensor(patch[None].copy())

        if self.transform is not None:
            patches = self.transform(patches)

        return patches, label, study_id


class ForaminalNarrowingDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "../../TotalSpineSeg",
        vol_paths: list = None,
        seg_paths: list = None,
        labels_csv: str = "./data/train.csv",
        transform: any = None,
        exclude: list = None,
    ):

        text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

        self.transform = transform
        self.root_dir = root_dir

        self.labels = pd.read_csv(labels_csv)
        self.labels = self.labels[
            [
                "study_id",
                "left_neural_foraminal_narrowing_l1_l2",
                "left_neural_foraminal_narrowing_l2_l3",
                "left_neural_foraminal_narrowing_l3_l4",
                "left_neural_foraminal_narrowing_l4_l5",
                "left_neural_foraminal_narrowing_l5_s1",
                "right_neural_foraminal_narrowing_l1_l2",
                "right_neural_foraminal_narrowing_l2_l3",
                "right_neural_foraminal_narrowing_l3_l4",
                "right_neural_foraminal_narrowing_l4_l5",
                "right_neural_foraminal_narrowing_l5_s1",
            ]
        ]

        rows_with_nan = self.labels[self.labels.isna().any(axis=1)]
        excludes = exclude + list(rows_with_nan.values[:, 0])
        self.labels = self.labels.dropna()
        self.labels = self.labels.replace(text2int)

        exclude_vol = []
        exclude_seg = []

        print(len(vol_paths))

        for study_id in excludes:
            for i in range(len(vol_paths)):
                if "sub-" + str(study_id) in vol_paths[i]:
                    exclude_vol.append(vol_paths[i])
                    exclude_seg.append(seg_paths[i])

        for x in exclude_vol:
            vol_paths.remove(x)

        for x in exclude_seg:
            seg_paths.remove(x)

        print(len(vol_paths))

        vol_paths.sort()
        seg_paths.sort()
        self.vol_paths = vol_paths
        self.seg_paths = seg_paths

    def __len__(self):
        return len(self.vol_paths)

    def __getitem__(self, idx):

        vol_path = self.vol_paths[idx]
        x = vol_path.split("/")[-1]
        x = x[:-7] + "_0000.nii.gz"

        study_id = x.split("_")[0][4:]

        seg_path = self.seg_paths[idx]

        label = (
            self.labels[self.labels["study_id"] == int(study_id)]
            .values[0, 1:]
            .astype(int)
        )

        vol = Image(self.root_dir + "/output/input/" + x)
        vol.change_orientation("LSA")
        vol = vol.data

        seg = Image(seg_path)
        seg = seg.change_orientation("LSA")
        seg = seg.data

        D, H, W = vol.shape
        discs = np.isin(seg, [202, 203, 204, 205, 206]).astype(int)
        disc_l5 = np.isin(seg, [202]).astype(int)
        disc_l4 = np.isin(seg, [203]).astype(int)
        disc_l3 = np.isin(seg, [204]).astype(int)
        disc_l2 = np.isin(seg, [205]).astype(int)
        disc_l1 = np.isin(seg, [206]).astype(int)
        # spinal_canal = np.isin(seg, [201]).astype(int)

        discs = [disc_l1, disc_l2, disc_l3, disc_l4, disc_l5]
        levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
        patches_left, patches_right = {}, {}

        w = 20
        for i, disc in enumerate(discs):
            patch_l, patch_r = patch_extraction2(vol, disc, d=16, h=40, w=20)
            patches_left[levels[i]] = torch.Tensor(patch_l[None].copy())
            patches_right[levels[i]] = torch.Tensor(patch_r[None].copy())

        if self.transform is not None:
            patches_left = self.transform(patches_left)
            patches_right = self.transform(patches_right)

        return patches_left, patches_right, label, study_id


class SubarticularStenosisDataset(Dataset):
    def __init__(
        self,
        vol_paths: list = None,
        labels_csv: str = "../data/train.csv",
        transform: any = None,
        exclude: list = [],
    ):

        text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

        self.transform = transform

        self.vol_paths = vol_paths

        self.labels = pd.read_csv(labels_csv)
        self.labels = self.labels[
            [
                "study_id",
                "right_subarticular_stenosis_l1_l2",
                "right_subarticular_stenosis_l2_l3",
                "right_subarticular_stenosis_l3_l4",
                "right_subarticular_stenosis_l4_l5",
                "right_subarticular_stenosis_l5_s1",
                "left_subarticular_stenosis_l1_l2",
                "left_subarticular_stenosis_l2_l3",
                "left_subarticular_stenosis_l3_l4",
                "left_subarticular_stenosis_l4_l5",
                "left_subarticular_stenosis_l5_s1",
            ]
        ]

        rows_with_nan = self.labels[self.labels.isna().any(axis=1)]
        excludes = exclude + list(rows_with_nan.values[:, 0])
        self.labels = self.labels.dropna()
        self.labels = self.labels.replace(text2int)

        exclude_vol = []

        for study_id in excludes:
            for i in range(len(self.vol_paths)):
                if "sub-" + str(study_id) in self.vol_paths[i]:
                    exclude_vol.append(self.vol_paths[i])

        for x in exclude_vol:
            self.vol_paths.remove(x)

        self.vol_paths.sort()

    def __len__(self):
        return len(self.vol_paths)

    def __getitem__(self, idx):

        vol_path = self.vol_paths[idx]
        x = vol_path.split("/")[-1]

        study_id = x.split("_")[0][4:]
        lvl = x.split("_")[-3] + "_" + x.split("_")[-2]
        lvl = lvl.lower()

        label = (
            self.labels[self.labels["study_id"] == int(study_id)][
                [
                    "left_subarticular_stenosis_" + lvl,
                    "right_subarticular_stenosis_" + lvl,
                ]
            ]
            .values[0]
            .astype(int)
        )

        vol = Image(vol_path)
        vol.change_orientation("ASL")
        vol = vol.data

        _, _, D = vol.shape

        patch_left, patch_right = vol[:, :, : D // 2], vol[:, :, D // 2 :]

        if self.transform is not None:
            patch_left = self.transform(patch_left[None].copy())
            patch_right = self.transform(patch_right[None].copy())

        return patch_left, patch_right, label, study_id
