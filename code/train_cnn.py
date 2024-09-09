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
    RocCurveDisplay,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    f1_score
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
    RandRotated,
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
    experiment,
    seqtype,
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

    C = 5

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
            
            try :
                patches, label, id = batch_data
            except RuntimeError:
                print(study_id)
            
            for i in range(C):    
                img = patches[i+1][0]
                # plt.figure(figsize=(10, 10))
                # plt.imshow(img[0,10])
                # plt.savefig("images/patch_level_{i}.png")
                
                # outputs, _ = model(patches[i+1].to(device)) 
                outputs = model(patches[i+1].to(device)) 

                loss += criterion(outputs, label[:, i].to(device))                
            epoch_len = len(train_data) // train_loader.batch_size
            total_loss = loss.item() / C
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
                    for i in range(C):    
                        # outputs, _ = model(patches[i+1].to(device))
                        outputs = model(patches[i+1].to(device)) 
                        loss += criterion(outputs[:], label[:, i].to(device))                
                    epoch_len = len(train_data) // train_loader.batch_size
                    total_loss = loss.item() / C
                    metric_eval += total_loss
                except torch.cuda.OutOfMemoryError:
                    print(id)
            metric_eval /= step 
            metric_values.append(metric_eval)

            if metric_eval < best_metric:
                best_metric = metric_eval
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), f"checkpoints/grading_network_experiment_{experiment}_{seqtype}.pth"
                )
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current metric: {metric_eval:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    torch.save(
            model.state_dict(), f"checkpoints/grading_network_experiment_{experiment}_last_epoch.pth"
                )

    # print(
    #     f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    # )
    writer_train.close()

    return exceptions, epoch_loss_values, metric_values

def test(model, test_loader, test_data, criterion, device, seqtype):
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    model.eval()
    num_correct = 0
    metric_count = 0
    total_loss = 0
    y_true = []
    y_pred = []

    C=5

    for batch_data in tqdm(test_loader):
        
        patches, label, id = batch_data
            
        with torch.no_grad():
            loss = 0
            for i in range(C):    
                # outputs, _ = model(patches[i+1].to(device)) 
                outputs = model(patches[i+1].to(device))
                loss += criterion(outputs[:], label[:, i].to(device))                

                # print(list(outputs.argmax(dim=-1).cpu().numpy()))
                y_pred += list(outputs.cpu().numpy())
                y_true += list(label[:,i].cpu().numpy())
                value = torch.eq(outputs.argmax(dim=-1).cpu(), label[:, i])
                metric_count += len(value)
                num_correct += value.sum().item()
                      
                # print(outputs)
            total_loss += loss.item() / 5

    metric = num_correct / metric_count
    epoch_len = len(test_data) // test_loader.batch_size
    total_loss /= epoch_len

    return metric, y_true, y_pred, total_loss


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
    weight_decay = config["weight_decay"]
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
        "left-sag-T1": ["left_neural_foraminal_narrowing",],
        "right-sag-T1" : ["right_neural_foraminal_narrowing"],
        "sag-T2": ["spinal_canal_stenosis"],
        "left-ax-T2": ["left_subarticular_stenosis"],
        "right-ax-T2": ["right_subarticular_stenosis"]
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
    val_id, test_id = train_test_split(val_id, test_size=0.5, random_state=42)

    # subjects to exclude

    exclude = list(np.load(exclude_file))

    # Build dataset and dataloader
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    description = pd.read_csv("./data/train_series_descriptions.csv")
    
    exclude = exclude + [
        2669990440,
        2830065820,
        3493057494,
        404602713,
        4646740,
        3507232306,
        933559951,
        1261271580,
        2780918205,
        228290246,
        4112621380,
        4096820034,
        391103067,
        3835824946,
        85480902,
        183230492,
        2280737054,
        3429409220,
        3808402167,
        2661178959,
        3242507143,
        652905669,
        3128832901,
        1573051559,
        3429410502,
        1289563234,
        2560914880,
        3966348292,
        3039901962,
        3532229145,
        1334624436,
        2056309275,
        1452809491,
        2294509371,
        4165566893,
        2496267917,
        2814554321,
        2091088734,
        2795583238,
        3542358517,
        3240785276, 
        3817394595, 
        3029953735, 
        4017932238, 
        1670838975, 
        1178209527, 
        1906657742,
                    1805675557,
                    2925530521,
                    2361533111,
                    4231198665,
                    3489581041,
                    2925530521,
                    1723430291,
                    2447825792,
                    2885881158,
                    4271960965,
                    2754246172,
                    1288134514,
                    886995462,
                    1835489622,
                    3480260143,
                    289846404,
                    416503281,
                    1879308612,
                    347228139,
                    1028909382,
                    3836986623,
                    2411161648,
                    3032490582,
                    1104422628,
                    916362094,
                    2465173537,
                    3329250043,
                    4167935162,
                    1106510276,
                    3308442440,
                    1805675557,
                    450154999,
                    478913051,
                    3068678959,
                    1784445928,
                    1474322336,
                    1096630192,
                    1868615696,
                    1935490243,
                    3832874334,
                    624497208,
                    2627142799,
                    1199116491,
                    4205258367,
                    1133001306,
                    1378385941,
                    607371793,
                    2072240933,
                    376657226,
                    3684608097,
                    26342422,
                    3881903999,
                    372642770,
                    2109299850,
                    3759970625,
                    2826913245,
                    3068697362,
                    3824003946,
                    3559395900,
                    3234968622,
                    2815518245,
                    4279958262,
                    3068697362,
                    2826913245,
                    3731783147,
                    4173917544,
                    87937369,
                    3887124538,
                    3912497560,
                    3337564969,
                    723551942, 
                    765688458,
                    2794192602, 
                    765688458,
                    1190473557,
                    2616775351,
                    38281420,
                    1190473557,
                    286903519,
                    2480600394,
                    665627263,
                    1438760543,
                    568440982,
                    1510451897,
                    1880970480,
                    3138242355,
                    1901348744,
                    1850731145,
                    2151467507,
                    2316015842,
                    4072191052,
                    2444340715,
                    4227229807,
                    1245057921,
                    2151509334,
                    2905025904,
                    859570985,
                    1820866003,
                    2015704745,
                    283265383,
                    765688458,
                    1237708996,
                    3819260179,
                    52695609,
                    1973833645,
                    209512460,
                    3390414227,
                    10728036
               ]
    
    if seqtype.endswith("ax-T2"):
        # train_transform = None
        train_transform = Compose(
                [
                    # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                    NormalizeIntensityd(keys=[1, 2, 3, 4, 5], 
                                        nonzero=False, 
                                        channel_wise=False),
                    # Resized(keys=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    #     spatial_size=(96, 96)
                    # ),
                    ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(128, 128) #(64, 64)
                ),   
                RandGaussianSmoothd(keys=[1, 2, 3, 4, 5], prob=0.1)
                ]
        )
        val_transform = Compose(
                [
                    # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                    NormalizeIntensityd(keys=[1, 2, 3, 4, 5], 
                                        nonzero=False, 
                                        channel_wise=False),
                    # Resized(keys=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    #     spatial_size=(96, 96)
                    # ),
                    ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(128, 128) #(64, 64)
                ),   
                RandGaussianSmoothd(keys=[1, 2, 3, 4, 5], prob=0.1)
                ]
        )
    
    elif seqtype=="sag-T2":
        train_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=[1, 2, 3, 4, 5], 
                                    nonzero=False, 
                                    channel_wise=False),
                # Resized(keys=[1, 2, 3, 4, 5],
                #         spatial_size=(64, 128)
                # ),
                ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(3, 64, 128)
                ),
                RandGaussianSmoothd(keys=[1, 2, 3, 4, 5], prob=0.2),
                RandRotated(keys=[1, 2, 3, 4, 5], prob=0.2, range_x = 0.5, range_y = 0.5, range_z = 0.5)
            ]
        )
        val_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10], 
                                    nonzero=False, 
                                    channel_wise=False),
                ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(3, 64, 128)
                ),
                # ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10],
                #         spatial_size=(64, 128)
                # )                
            ]
        )
        
    elif seqtype==("left-sag-T1"):
        train_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10], 
                                    nonzero=False, 
                                    channel_wise=False),
                ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(3, 64, 128)
                ),
                # ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10],
                #         spatial_size=(64, 128)
                # )                
                RandGaussianSmoothd(keys=[1, 2, 3, 4, 5], prob=0.2),
                RandRotated(keys=[1, 2, 3, 4, 5], prob=0.2, range_x = 0.5, range_y = 0.5, range_z = 0.5)
            ]
        )
        val_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10], 
                                    nonzero=False, 
                                    channel_wise=False),
                ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(3, 64, 128)
                ),
                # ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10],
                #         spatial_size=(64, 128)
                # )                
            ]
        )
        
    elif seqtype==("right-sag-T1"):
        train_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10], 
                                    nonzero=False, 
                                    channel_wise=False),
                ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(3, 64, 128)
                ),
                # ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10],
                #         spatial_size=(64, 128)
                # )                
            ]
        )
        val_transform = Compose(
            [
                # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
                NormalizeIntensityd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10], 
                                    nonzero=False, 
                                    channel_wise=False),
                ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5],
                        spatial_size=(3, 64, 128)
                ),
                # ResizeWithPadOrCropd(keys=[1, 2, 3, 4, 5], #, 6, 7, 8, 9, 10],
                #         spatial_size=(64, 128)
                # )                
            ]
        )
        
    # elif seqtype=="sag-T1":
    #     train_transform = Compose(
    #     [
    #         # EnsureChannelFirstd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
    #         NormalizeIntensityd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"], 
    #                             nonzero=False, 
    #                             channel_wise=False),
    #         Resized(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"],
    #                 spatial_size=(64, 128)
    #         ),
    #         # RandGaussianSmoothd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"], prob=0.1),
            
    #     ]
    # )

    
    print(seqtype)
    
    all_data = RSNAPatchDataset(
                root_dir = "./data/train_images/",
                study_ids = study_ids,
                seqtype = seqtype,
                description = description,
                coordinates = coordinates, 
                labels = id_label,
                exclude = exclude,
                transform = train_transform
    )
    
    train_data = RSNAPatchDataset(
                root_dir = "./data/train_images/",
                study_ids = train_id,
                seqtype = seqtype,
                description = description,
                coordinates = coordinates, 
                labels = id_label,
                exclude = exclude,
                transform = train_transform
    )
    val_data = RSNAPatchDataset(
            root_dir = "./data/train_images/",
            study_ids = val_id,
            seqtype = seqtype,
            description = description,
            coordinates = coordinates, 
            labels = id_label,
            exclude = exclude,
            transform = val_transform
    )
    test_data = RSNAPatchDataset(
            root_dir = "./data/train_images/",
            study_ids = test_id,
            seqtype = seqtype,
            description = description,
            coordinates = coordinates, 
            labels = id_label,
            exclude = exclude,
            transform = val_transform
    )

    print("Length of train dataset :", len(train_data))
    print("Length of validation dataset :", len(val_data))
    # print("Length of test dataset :", len(test_data))

    all_data_loader = DataLoader(dataset=all_data, batch_size=1, shuffle=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

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
    #                 spatial_dims=2,
    #                 n_input_channels=1,
    #                 num_classes=3,
    #             ).to(device)
    
    # ViT
    
    # model = ViT(in_channels=1, 
    #             img_size=(64,128), 
    #             patch_size=(64,64),
    #             spatial_dims=2,
    #             pos_embed='conv', 
    #             post_activation = None,
    #             classification=True,
    #             num_classes=3
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # scheduler = CosineAnnealingLR(optimizer, T_max=50)
    torch.manual_seed(seed=42)

    # L = []

    # for batch_data in tqdm(all_data_loader):
    #     patches, label, study_id = batch_data

        # print(patches[1].shape)
             
    
    # print(L)
    # print(len(L))
  
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
            experiment,
            seqtype,
            val_interval=val_interval
        )
        
        plt.figure()
        plt.plot(list(range(epochs)), epoch_loss_values)
        plt.plot([val_interval*i for i in range(epochs//val_interval)], metric_values)
        plt.legend(["training", "validation"])
        plt.savefig(f"Training_losses_{seqtype}.png")
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

    model = ResNet(
            block="basic",
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=3,
    ).to(device)
    
    # model = ViT(in_channels=1, 
    #             img_size=(64,128), 
    #             patch_size=(64,64),
    #             spatial_dims=2,
    #             pos_embed='conv', 
    #             post_activation = None,
    #             classification=True,
    #             num_classes=3
    #             ).to(device)
    
    # model = ResNet(
    #             block="basic",
    #             layers=[2, 2, 2, 2],
    #             block_inplanes=[64, 128, 256, 512],
    #             spatial_dims=2,
    #             n_input_channels=1,
    #             num_classes=3,
    #         ).to(device)

    
    
    print(f"checkpoints/grading_network_experiment_{experiment}_{seqtype}.pth")
    model.load_state_dict(torch.load(f"checkpoints/grading_network_experiment_{experiment}_{seqtype}.pth"))
    model.eval()
    
    metric, y_true, y_pred, total_loss = test(model=model, 
                                  test_loader=test_loader, 
                                  test_data = test_data, 
                                  criterion = criterion,
                                  device = device,
                                  seqtype = seqtype)

    print("Total loss : {:.2f}".format(total_loss))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # n, k = y_pred.shape    
    
    np.save("y_true.npy", y_true)
    np.save("y_pred.npy", y_pred)

    y_test = y_true #.reshape(n*k)
    y_pred = y_pred #.reshape(n*k)

    print(y_pred.shape)
    print(y_true.shape)
    pred = y_pred.argmax(axis=-1)    
    
    print("Raw accuracy score on test set :", metric)
    print("Balanced accuracy score on test set : {:.2f}".format(balanced_accuracy_score(y_true=y_test, y_pred=pred)))
    print("F1 score on test set : {:.2f}".format(f1_score(y_true=y_test, y_pred=pred, average="macro")))
    

    labels = [0, 1, 2]
    cm = confusion_matrix(y_test, pred, labels=labels)
    print(type(cm))
    print("confusion matrix :", np.array(cm))
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = cm / row_sums
    

    plt.imshow(cm, cmap="bwr")
    plt.colorbar()
    plt.savefig(f"normalized_confusion_matrix_{seqtype}.png")
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()
    
    plt.imshow(cm)
    plt.savefig(f"confusion_matrix_{seqtype}.png")
    plt.show()
    
    
    
    # fpr, tpr, thresholds = roc_curve(y_true, 1-y_pred)
    # roc_auc = auc(fpr, tpr)
    # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                                 estimator_name='CNN')

    # display.plot()
    # plt.savefig("roc.png")
    # plt.show()


if __name__ == "__main__":
    main()
