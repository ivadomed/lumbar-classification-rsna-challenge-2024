"""
This Python script aims at training UNet model for the
RSNA 2024 kaggle data challenge. 

Author : Simon Queric
2024-07-11

"""

import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from monai.data import DataLoader
from dataset import RSNADataset, UNetDataset
import torch
from torch.nn.functional import cross_entropy
import argparse
from monai.transforms import (
    LoadImage,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    Compose,
    ResizeWithPadOrCropd,
    CenterScaleCropd,
    RandFlipd,
    Resized,
    NormalizeIntensityd,
    GaussianSmoothd,
)
from monai.losses import DiceLoss
from monai.networks.nets import ResNet, EfficientNetBN, ViT, BasicUNet, DynUNet

from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

# class DiceLoss(torch.nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#         self.sigmoid = torch.nn.Sigmoid()
#     def forward(self, x, y):
#         x = self.sigmoid(x)
#         inter = (x*y).sum()
#         union = x.sum() + y.sum()
#         dice = (2*inter+1)/(1+union)
#         return 1 - dice
    
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

def merge_labels(batch_data : dict):
    _, w, h, d = batch_data["L1/L2"].shape

    
def train(
    model,
    epochs,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    train_data,
    val_loader,
    device,
    writer,
    autocast,
    scaler
):
    """
    This function train a model for the RSNA kaggle data challenge.
    """

    epoch_loss_values = []
    metric_values = []
    losses = []
    best_metric = torch.inf
    val_interval = 1

    exceptions = []

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            # try:
            inputs, labels, id = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["study_id"],
            )
            
            # B, C, D, H, W = inputs.shape
            # inputs = inputs.reshape((B*C*D, 1, H, W))
            optimizer.zero_grad()
            outputs = model(inputs) 
            # B, C, D, H, W = labels.shape
            # labels = labels.reshape((B*D, C, H, W))
            loss = criterion(outputs, labels)
            # loss.backward()
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_data) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, study id : {id}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            # except RuntimeError:
            #     print("Runtime error on subject ", batch_data["study_id"][0])
            # id = int(id)
            # print("inputs.shape :", inputs.shape)
            # print("labels.shape :", labels.shape)
            
            # except RuntimeError:
            #     print("study id :", id)
            #     print("Volume shape :", inputs.shape)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if scheduler is not None:
            scheduler.step()
        
        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        print("Total norm of model parameters gradients epoch {} : {:.3e}".format(epoch+1, total_norm))
        
        if (epoch + 1) % val_interval == 0:
            model.eval()

            # negative log loss for validation
            metric_eval = 0
            step = 0
            for batch_data in tqdm(val_loader):
                step+=1
                inputs, labels, id = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                    batch_data["study_id"],
                )
                with torch.no_grad():
                    
                    try:
                        # B, C, D, H, W = inputs.shape
                        # inputs = inputs.reshape((B*C*D, 1, H, W))
                        outputs = model(inputs) 
                        # B, C, D, H, W = labels.shape
                        # labels = labels.reshape((B*D, C, H, W))
                        loss = criterion(outputs, labels)
                        metric = loss.item()
                        metric_eval += metric
                        writer.add_scalar("val_loss", loss.item(), epoch_len * epoch + step)
                    except RuntimeError:
                        exceptions.append(id)
                        print("RuntimeError")
                    
                    
                    

            metric_eval /= step
            metric_values.append(metric_eval)

            if metric_eval < best_metric:
                best_metric = metric_eval
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), "best_metric_model_unet2.pth"
                )
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current metric: {metric_eval:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()

    return exceptions #, epoch_loss_values, metric_values


def test(model, test_loader, device):
    model.eval()
    num_correct = 0.0
    metric_count = 0
    y_true = []
    y_pred = []

    for batch_data in tqdm(test_loader):
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        with torch.no_grad():
            outputs = model(inputs)
            b, n = outputs.shape
            k = n // 3
            outputs = outputs.reshape((b, k, 3))
            y_pred += list(outputs.cpu().numpy().reshape((b, k, 3)).argmax(axis=-1))
            y_true += list(labels.cpu().numpy().reshape((b, k)))
            # print(y_pred, y_true)
            for c in range(k):
                value = torch.eq(
                    outputs[:, c].argmax(dim=-1), labels[:, c]
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
        config = yaml.safe_load(f)

    folder = config["folder"]
    gpu = config["gpu"]
    use_amp = bool(config["use_amp"])
    scaler = bool(config["scaler"])
    exclude_file = config["exclude-file"]
    
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")

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
    resize = config["resize"]
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
    

    cond_lev = ["study_id"]

    CONDITIONS = seq2cond[seqtype]

    for level in LEVELS:
        for cond in CONDITIONS:
            cond_lev.append(cond + "_" + level)

    # print(cond_lev)
    id_label = id_label[cond_lev]
    id_label = id_label.dropna()
    id_label = id_label.replace(text2int)
    study_ids = id_label.values[:, 0].astype(int)  # store id of each subject

    print(study_ids)

    # Train - Validation - Test split
    # 0.5 - 0.25 - 0.25
    train_id, val_id = train_test_split(study_ids, test_size=0.3, random_state=42)
    val_id, test_id = train_test_split(val_id, test_size=0.5, random_state=42)
    # train_id  =  [4003253]
    # val_id = val_id[0:1]
    # subject to exclude

    exclude = list(np.load(exclude_file))

    # Transforms

    transform = Compose(
        [
            Resized(keys=["image", "label"],
                    spatial_size=resize,
                    mode = ["area", "nearest"]
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ]
    )

    # Build dataset and dataloader

    
    train_data = UNetDataset(
        root_dir=folder,
        study_ids=train_id,
        seqtype=seqtype,
        coordinates=coordinates,
        cond = "Left Neural Foraminal Narrowing",
        exclude=exclude,
        transform=transform,
    )
    
    val_data = UNetDataset(
        root_dir=folder,
        study_ids=val_id,
        seqtype=seqtype,
        coordinates=coordinates,
        cond = "Left Neural Foraminal Narrowing",
        exclude=exclude,
        transform=transform,
    )
    test_data = UNetDataset(
        root_dir=folder,
        study_ids=test_id,
        seqtype=seqtype,
        coordinates=coordinates,
        cond = "Left Neural Foraminal Narrowing",
        exclude=exclude,
        transform=transform,
    )
    print("Length of train dataset :", len(train_data))
    print("Length of validation dataset :", len(val_data))
    print("Length of test dataset :", len(test_data))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter()

    device = torch.device(f"{gpu}" if torch.cuda.is_available() else "cpu")

    print("device =", device)
    
    autocast = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler, init_scale=4096)

    model = BasicUNet(spatial_dims=3, 
                      in_channels=1, 
                      out_channels=5
                      ).to(device)
    
    # model = DynUNet(spatial_dims=3,
    #             in_channels=1,
    #             out_channels=5,
    #             kernel_size=(16, 32, 64, 128, 256),
    #             strides=(2, 2, 2, 2),
    #             upsample_kernel_size=(2, 2, 2)
    #         ).to(device)
    
    
    # Optimization method, loss criterion
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weigth).to(device))
    
    class DiceCELoss(torch.nn.Module):
        def __init__(self):
            super(DiceCELoss, self).__init__()
            self.sigmoid = torch.nn.Sigmoid()
            self.cross_entropy = torch.nn.CrossEntropyLoss()
        
        def forward(self, x, y):
            x2 = x.softmax(dim=1) #self.sigmoid(x)
            inter = (x2*y).sum()
            union = x2.sum() + y.sum()
            dice_loss = 1 - (2*inter+1e-3)/(1e-3+union)
            ce_loss = self.cross_entropy(x, y)
            return ce_loss # dice_loss + ce_loss

    # class CustomLoss(torch.nn.Module):
    #     def __init__(self):
    #         super(CustomLoss, self).__init__()
    #         self.dice = DiceLoss(sigmoid=False)
    #         self.mse = torch.nn.MSELoss()
    #     def forward(self, x, y):
    #         dice = self.dice(x, y)
    #         mse = self.mse(x, y)
    #         return dice + mse
    
    criterion = DiceLoss(sigmoid=True) 
    # criterion = DiceCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = None # lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # scheduler = CosineAnnealingLR(optimizer, T_max=50)
  
    if epochs > 0:
        exceptions, epoch_loss_values, metric_values = train(
            model,
            epochs,
            optimizer,
            criterion,
            scheduler,
            train_loader,
            train_data,
            val_loader,
            device,
            writer,
            autocast,
            scaler
        )
        
        plt.figure()
        plt.plot(list(range(epochs)), epoch_loss_values)
        plt.savefig("Training_losses.png")
        plt.figure()
        plt.plot(list(range(epochs)), metric_values)
        plt.savefig("Validation_losses.png")

        # Save losses, evaluation metric

        # np.save("epoch-loss-values.npy", np.array(epoch_loss_values))
        # np.save("metric_values.npy", np.array(metric_values))

        print(exceptions)
    # exceptions = np.array(exceptions)
    # np.save("exceptions.npy", exceptions)

    shapes = {}

    model.load_state_dict(torch.load("best_metric_model_unet2.pth"))
    model.eval()
    metric, y_true, y_pred = test(model=model, test_loader=test_loader, device=device)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n, k = y_pred.shape
    
    
    np.save("y_true.npy", y_true)
    np.save("y_pred.npy", y_pred)

    y_test = y_true.reshape(n*k)
    y_pred = y_pred.reshape(n*k)

    print(y_pred.shape)
    print(y_true.shape)
    # pred = y_pred.argmax(axis=-1)
    
    print("Raw accuracy score on test set :", metric)

    labels = [0, 1, 2]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
