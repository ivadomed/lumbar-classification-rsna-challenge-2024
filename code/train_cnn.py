"""
This Python script aims at training classification models for the
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
    roc_curve,
    auc,
    RocCurveDisplay
)
from monai.data import DataLoader
from utils import *
from dataset import *
import torch
from torch.nn.functional import cross_entropy
import argparse
from monai.transforms import (
    RandGaussianSmoothd,
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
from monai.networks.nets import ResNet, EfficientNetBN, ViT

from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR

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

def train(
    model,
    epochs,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    train_data,
    val_loader,
    val_data,
    device,
    writer_train,
    autocast,
    scaler,
    val_interval
):
    """
    This function train a model for the RSNA kaggle data challenge.
    """

    epoch_loss_values = []
    metric_values = []
    losses = []
    best_metric = torch.inf

    exceptions = []

    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            total_loss = 0
            optimizer.zero_grad()
            loss = 0 
            # try:
            patches, label, id = batch_data
            for i in range(5):    
                # id = int(id)        
                outputs = model(patches[LEVELS[i]].to(device)) 
                loss += criterion(outputs, label[:, i].to(device))                
            epoch_len = len(train_data) // train_loader.batch_size
            total_loss = loss.item() / 5
            epoch_loss += total_loss
            scaler.scale(loss).backward()
            optimizer.step()
            # print(f"{step}/{epoch_len}, train_loss: {total_loss:.4f}, study id : {id}")
            writer_train.add_scalar("train_loss", total_loss, epoch_len * epoch + step)
            # except RuntimeError as e:
            #     print("Runtime error on subéject ", id)
            # except TypeError as e:
            #     print(id)
            # except FileNotFoundError as e:
            #     pass
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            metric_eval = 0
            step = 0
            
            for batch_data in tqdm(val_loader):
                step += 1
                total_loss = 0
                patches, label, id = batch_data
                try:
                    loss=0
                    for i in range(5):    
                        outputs = model(patches[LEVELS[i]].to(device)) 
                        loss += criterion(outputs[:], label[:, i].to(device))                
                    epoch_len = len(train_data) // train_loader.batch_size
                    total_loss = loss.item() / 5
                    metric_eval += total_loss
                except torch.cuda.OutOfMemoryError:
                    print(id)
            metric_eval /= step 
            metric_values.append(metric_eval)

            if metric_eval < best_metric:
                best_metric = metric_eval
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), f"best_metric_model_classification3d_array.pth"
                )
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current metric: {metric_eval:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            

    # print(
    #     f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    # )
    writer_train.close()

    return exceptions, epoch_loss_values, metric_values


def test(models, test_loader, device):
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    for i in range(5):
        models[i].eval()
    num_correct = 0.0
    metric_count = 0
    y_true = []
    y_pred = []

    for batch_data in tqdm(test_loader):
        for i in range(5):
            inputs, labels, id = (
                batch_data[LEVELS[i]].to(device),
                batch_data["label"].to(device),
                batch_data["study_id"][0],
            )
            with torch.no_grad():
                outputs = models[i](inputs[None])
                y_pred += list(outputs.cpu().numpy()[:,0])
                y_true += list(labels[:,i].cpu().numpy())
                # print(y_pred, y_true)
                value = torch.eq(outputs.argmax(dim=-1), labels[:, i])
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
    experiment = config["experiment"]
    use_amp = bool(config["use_amp"])
    scaler = bool(config["scaler"])
    exclude_file = config["exclude_file"]
    
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
    val_interval = config["val_interval"]

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
    train_id, val_id = train_test_split(study_ids, test_size=0.5, random_state=42)
    # val_id, test_id = train_test_split(val_id, test_size=0.5, random_state=42)

    # subject to exclude

    exclude = list(np.load(exclude_file))

    # Transforms

    if type(crop_padd_size)==list:
        transform = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes=orientation),
                Spacingd(
                    keys=["image"],
                    pixdim=pixdim,
                    mode=interp_mode,
                ),
                Resized(keys=["image"],
                        spatial_size=resize
                ),
                # ResizeWithPadOrCropd(keys=["image"], spatial_size=crop_padd_size),
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            ]
        )

    elif eval(crop_padd_size) is None:
        transform = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes=orientation),
                Spacingd(
                    keys=["image"],
                    pixdim=pixdim,
                    mode=interp_mode,
                ),
                Resized(keys=["image"],
                        spatial_size=resize
                ),
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            ]
        )

    # Build dataset and dataloader
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    description = pd.read_csv("./data/train_series_descriptions.csv")
    
    exclude = exclude + [
                    723551942, 
                    765688458,
                    2794192602, 
                    765688458,
                    1190473557,
                    38281420,
                    1190473557,
                    286903519,
                    2480600394,
                    665627263,
                    1438760543,
                    1510451897,
                    1880970480,
                    1901348744,
                    2151467507,
                    2316015842,
                    2444340715,
                    2151509334,
                    2905025904,
                    1820866003,
                    283265383,
                    765688458,
                    3819260179,
                    1973833645
               ]
    
    train_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"], 
                                    nonzero=False, 
                                    channel_wise=False),
                ResizeWithPadOrCropd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"],
                        spatial_size=(20, 60, 150)
                ),
                RandGaussianSmoothd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"], prob=0.1),
                
            ]
        )
    
    train_data = RSNAPatchDataset(
                root_dir = "./data/train_images/",
                study_ids = train_id,
                contrast = "t2",
                description = description,
                coordinates = coordinates, 
                labels = id_label,
                exclude = exclude,
                transform = train_transform
    )
    val_data = RSNAPatchDataset(
            root_dir = "./data/train_images/",
            study_ids = val_id,
            contrast = "t2",
            description = description,
            coordinates = coordinates, 
            labels = id_label,
            exclude = exclude,
            transform = train_transform
    )
    # test_data = RSNAPatchDataset(
    #         root_dir = "./data/train_images/",
    #         study_ids = test_id,
    #         contrast = "t2",
    #         description = description,
    #         coordinates = coordinates, 
    #         labels = id_label,
    #         exclude = exclude,
    #         transform = train_transform
    # )

    print("Length of train dataset :", len(train_data))
    print("Length of validation dataset :", len(val_data))
    # print("Length of test dataset :", len(test_data))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # for batch_data in tqdm(train_loader):
    #     patches, label, id = batch_data
    #     print("-"*10)
    #     print(id)
    #     print("-"*10)

    writer_train = SummaryWriter("./runs/experiment_ ")
    # writer_val = SummaryWriter("Validation loss")

    device = torch.device(f"{gpu}" if torch.cuda.is_available() else "cpu")

    print("device =", device)
    
    autocast = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler, init_scale=4096)

    # ResNet10
    # model = ResNet(
    #                 block="basic",
    #                 layers=[1, 1, 1, 1],
    #                 block_inplanes=[64, 128, 256, 512],
    #                 spatial_dims=3,
    #                 n_input_channels=1,
    #                 num_classes=3,
    #             ).to(device)

    # ResNet17
    # model = ResNet(
    #                 block="basic",
    #                 layers=[2, 2, 2, 2],
    #                 block_inplanes=[64, 128, 256, 512],
    #                 spatial_dims=3,
    #                 n_input_channels=1,
    #                 num_classes=3,
    #             ).to(device)
    
    # ResNet34
    model = ResNet(
                    block="basic",
                    layers=[3, 4, 6, 3],
                    block_inplanes=[64, 128, 256, 512],
                    spatial_dims=3,
                    n_input_channels=1,
                    num_classes=3,
                ).to(device)
    
    inputs = torch.zeros((32, 64, 128)) 
    # writer_train.add_graph(model, inputs)

    
    
    # Models, optimization method, loss criterion
    
    # models = []
    # criterions = []
    # optimizers = [] 
    # for i in range(5):
    #     models.append(ResNet(
    #                 block="basic",
    #                 layers=[1, 1, 1, 1],
    #                 block_inplanes=[64, 128, 256, 512],
    #                 spatial_dims=3,
    #                 n_input_channels=1,
    #                 num_classes=2,
    #             ).to(device))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1., 2., 4.]).to(device), reduction="mean")
    # criterions.append(torch.nn.CrossEntropyLoss(weight=torch.Tensor(weigth).to(device)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-5)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # scheduler = CosineAnnealingLR(optimizer, T_max=50)
    torch.manual_seed(seed=42)
  
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
            val_data,
            device,
            writer_train,
            autocast,
            scaler,
            val_interval=val_interval
        )
        
        plt.figure()
        plt.plot(list(range(epochs)), epoch_loss_values)
        plt.plot([val_interval*i for i in range(epochs//val_interval)], metric_values)
        plt.legend(["training", "validation"])
        plt.savefig(f"Training_losses_{experiment}.png")
        plt.figure()
        
        plt.figure()
        plt.plot(list(range(epochs)), epoch_loss_values)
        plt.legend(["training"])
        plt.savefig(f"Train_{experiment}.png")
        plt.figure()
        
        plt.figure()
        plt.plot([val_interval*i for i in range(epochs//val_interval)], metric_values)
        plt.legend(["val"])
        plt.savefig(f"Val_{experiment}.png")
        plt.figure()
        
        # Save losses, evaluation metric

        # np.save("epoch-loss-values.npy", np.array(epoch_loss_values))
        # np.save("metric_values.npy", np.array(metric_values))

        print(exceptions)
    # exceptions = np.array(exceptions)
    # np.save("exceptions.npy", exceptions)

    # for i in range(5):
    #     models.append(ResNet(
    #                 block="basic",
    #                 layers=[1, 1, 1, 1],
    #                 block_inplanes=[64, 128, 256, 512],
    #                 spatial_dims=3,
    #                 n_input_channels=1,
    #                 num_classes=3))
    #     models[i].load_state_dict(torch.load(f"best_metric_model_classification3d_{i}_array.pth"))
    #     models[i].eval()
    
    # metric, y_true, y_pred = test(models=models, test_loader=test_loader, device=device)

    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)

    # # n, k = y_pred.shape
    
    
    # np.save("y_true.npy", y_true)
    # np.save("y_pred.npy", y_pred)

    # y_test = y_true #.reshape(n*k)
    # y_pred = y_pred #.reshape(n*k)

    # print(y_pred.shape)
    # print(y_true.shape)
    # # pred = y_pred.argmax(axis=-1)
    
    # print("Raw accuracy score on test set :", metric)

    # labels = [0, 1]
    # cm = confusion_matrix(y_test, y_pred<0.5, labels=labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # disp.plot()
    # plt.savefig("confusion_matrix.png")
    # plt.show()
    
    # fpr, tpr, thresholds = roc_curve(y_true, 1-y_pred)
    # roc_auc = auc(fpr, tpr)
    # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                                 estimator_name='CNN')

    # display.plot()
    # plt.savefig("roc.png")
    # plt.show()


if __name__ == "__main__":
    main()
