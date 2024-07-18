"""
This Python script aims at training classification models for the
RSNA 2024 kaggle data challenge. 

Author : Simon Queric
2024-07-11

"""

import sys
import json

sys.path.insert(0, "./code/")
import numpy as np
import pandas as pd
from tqdm import tqdm
from image import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from monai.data import DataLoader
from dataset import RSNADataset
import torch
from torch.nn.functional import cross_entropy
import argparse
from monai.transforms import (
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    ResizeWithPadOrCropd,
    CenterScaleCropd,
    RandFlipd,
    NormalizeIntensityd,
    GaussianSmoothd,
)

from monai.networks.nets import ResNet, EfficientNetBN, ViT
from torch.utils.tensorboard import SummaryWriter


def get_parser():
    parser = argparse.ArgumentParser(
        prog="rsna-2024-training",
        description="Train a classification model on rsna dataset",
        epilog="",
    )

    parser.add_argument(
        "-config", type=str, required=True, help="config file for training."
    )

    return parser


def get_shapes(loader, seqtype):
    """
    This function run through a data loader of RSNADataset to get shapes of volumes.
    The batch size of the data loader must be equal to one.
    """
    shapes = {}
    for batch_data in tqdm(loader):
        inputs, id = batch_data[0], batch_data[2][0]
        shapes[str(id)] = inputs.shape

    with open("shapes" + seqtype + ".json", "w") as f:
        json.dump(shapes, f)

    return


def train(
    model,
    epochs,
    optimizer,
    criterion,
    train_loader,
    train_data,
    val_loader,
    device,
    writer,
):
    """
    This function train a model for the RSNA kaggle data challenge.
    """

    epoch_loss_values = []
    metric_values = []
    losses = []
    best_metric = torch.inf
    val_interval = 2

    exceptions = []

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels, id = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["study_id"][0],
            )
            id = int(id)
            # print("inputs.shape :", inputs.shape)
            # print("labels.shape :", labels.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            b, n = outputs.shape
            k = n // 3
            outputs = outputs.reshape((b, k, 3))
            # print(outputs.shape)
            loss = 0
            _, c, _ = labels.shape
            for i in range(c):
                loss += (
                    criterion(outputs[:, i, :], labels[:, i, :]) / c
                )  # iterating across each level and condition
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_data) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, study id : {id}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            # negative log loss for validation
            metric_eval = 0
            step = 0
            for batch_data in tqdm(val_loader):
                step += 1
                inputs, labels, id = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                    batch_data["study_id"],
                )
                with torch.no_grad():
                    try:
                        outputs = model(inputs)
                        b, n = outputs.shape
                        k = n // 3
                        outputs = outputs.reshape((b, k, 3))
                    except RuntimeError:
                        exceptions.append(id)
                        print("RuntimeError")
                    metric = 0
                    _, c, _ = labels.shape
                    for i in range(c):
                        loss = cross_entropy(outputs[:, i, :], labels[:, i, :]) / c
                        metric += loss
                    metric_eval += metric

            metric_eval /= step
            metric_values.append(metric_eval)

            if metric_eval < best_metric:
                best_metric = metric_eval
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), "best_metric_model_classification3d_array.pth"
                )
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current metric: {metric_eval:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()

    return exceptions, epoch_loss_values, metric_values


def test(model, test_loader, device):
    model.eval()
    num_correct = 0.0
    metric_count = 0
    y_true = []
    y_pred = []

    for batch_data in tqdm(test_loader):
        inputs, labels, id = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
            batch_data["study_id"][0],
        )
        with torch.no_grad():
            outputs = model(inputs)
            b, n = outputs.shape
            k = n // 3
            outputs = outputs.reshape((b, k, 3))
            y_pred += list(outputs.cpu().numpy().reshape((b*k, 3)))
            y_true += list(labels.cpu().numpy().reshape((b*k, 3)))
            for c in range(k):
                value = torch.eq(
                    outputs[:, c].argmax(dim=-1), labels[:, c].argmax(dim=-1)
                )
                metric_count += len(value)
                num_correct += value.sum().item()

    metric = num_correct / metric_count

    return metric, y_true, y_pred


def main():

    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, "r") as f:
        config = json.load(f)

    folder = config["folder"]
    gpu = config["gpu"]

    # Hyperparameters
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    weigth = config["weights"]

    # Sequence type
    seqtype = config["seqtype"]

    # Transform parameters
    orientation = config["orientation"]
    pixdim = config["pixdim"]
    interp_mode = config["interpmode"]
    crop_padd_size = config["crop_padd_size"]
    LEVELS = config["levels"]

    # Dictionary mapping sequence type (contrast + orientation) to the associated condition.

    seq2cond = {
        "sag-T1": [
            "left_neural_foraminal_narrowing",
            "right_neural_foraminal_narrowing",
        ],
        "sag-T2": ["spinal_canal_stenosis"],
        "ax-T2": ["left_subarticular_stenosis", "right_subarticular_stenosis"],
    }

    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    id_label = pd.read_csv("./data/train.csv")
    id_label = id_label.replace(text2int)

    cond_lev = ["study_id"]

    CONDITIONS = seq2cond[seqtype]

    for level in LEVELS:
        for cond in CONDITIONS:
            cond_lev.append(cond + "_" + level)

    # print(cond_lev)
    id_label = id_label[cond_lev]
    study_ids = id_label.values[:, 0].astype(int)  # store id of each subject

    print(study_ids)

    # Train - Validation - Test split

    train_id, val_id = train_test_split(study_ids, test_size=1500, random_state=42)
    val_id, test_id = train_test_split(val_id, test_size=0.9, random_state=42)

    # subject to exclude

    exclude = list(np.load("exclude.npy"))

    # Transforms

    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes=orientation),  # RSP --> LIA
            Spacingd(
                keys=["image"],
                pixdim=pixdim,
                mode=interp_mode,
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_padd_size),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    # Build dataset and dataloader
    train_data = RSNADataset(
        root_dir=folder,
        study_ids=train_id,
        seqtype=seqtype,
        label_df=id_label,
        exclude=exclude,
        transform=transform,
    )
    val_data = RSNADataset(
        root_dir=folder,
        study_ids=val_id,
        seqtype=seqtype,
        label_df=id_label,
        exclude=exclude,
        transform=transform,
    )
    test_data = RSNADataset(
        root_dir=folder,
        study_ids=test_id,
        seqtype=seqtype,
        label_df=id_label,
        exclude=exclude,
        transform=transform,
    )

    print("Length of test dataset :", len(test_data))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)

    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    writer = SummaryWriter()

    device = torch.device(f"{gpu}" if torch.cuda.is_available() else "cpu")

    print("device =", device)

    model = ResNet(
        block="basic",
        layers=[1, 1, 1, 1],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=(len(cond_lev) - 1) * 3,
    ).to(device)

    # model = EfficientNetBN("efficientnet-b7",
    #                         spatial_dims=3,
    #                         in_channels=1,
    #                         num_classes=(len(cond_lev)-1)*3).to(device)

    # Optimization method, loss criterion
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weigth).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if epochs > 0:
        exceptions, epoch_loss_values, metric_values = train(
            model,
            epochs,
            optimizer,
            criterion,
            train_loader,
            train_data,
            val_loader,
            device,
            writer,
        )

        # Save losses, evaluation metric

        # np.save("epoch-loss-values.npy", np.array(epoch_loss_values))
        # np.save("metric_values.npy", np.array(metric_values))

        print(exceptions)
    # exceptions = np.array(exceptions)
    # np.save("exceptions.npy", exceptions)

    shapes = {}

    model.load_state_dict(torch.load("best_metric_model_classification3d_array.pth"))
    model.eval()
    metric, y_true, y_pred = test(model=model, test_loader=test_loader, device=device)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    np.save("y_true.npy", y_true)
    np.save("y_pred.npy", y_pred)

    print(y_true.shape)

    y_test = y_true.argmax(axis=-1)
    pred = y_pred.argmax(axis=-1)

    print("Raw accuracy score on test set :", metric)

    labels = [0, 1, 2]
    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
