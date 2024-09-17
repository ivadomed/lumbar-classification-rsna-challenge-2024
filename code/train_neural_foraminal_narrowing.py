"""
This Python script aims at training classification models for the
RSNA 2024 kaggle data challenge. 

We focus on Neural Foraminal Narrowing diagnosis.

Author : Simon Queric
2024-09-10

"""

import sys
import yaml
import numpy as np
import nibabel as nib
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
        levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

        for batch_data in tqdm(train_loader):
            step += 1
            total_loss = 0
            optimizer.zero_grad()
            loss = 0 
            
            try :
                patches_left, patches_right, label, study_id = batch_data
                
                for i in range(C):    
                
                    outputs = model(patches_left[levels[i]].to(device)) 
                    loss += criterion(outputs, label[:, i].to(device))
                for i in range(C):    
               
                    outputs = model(patches_right[levels[i]].to(device)) 
                    loss += criterion(outputs, label[:, 5+i].to(device)) 
                                   
                epoch_len = len(train_data) // train_loader.batch_size
                total_loss = loss.item() / (2*C)
                epoch_loss += total_loss
                scaler.scale(loss).backward()
                optimizer.step()
                # print(f"{step}/{epoch_len}, train_loss: {total_loss:.4f}, study id : {id}")
                writer_train.add_scalar("train_loss", total_loss, epoch_len * epoch + step)

            except Exception as e:
                print(f"Exception {e}")
            
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
                
                try:
                    patches_left, patches_right, label, study_id = batch_data
                    loss=0
                    for i in range(C):    
                        # outputs, _ = model(patches[levels[i]].to(device))
                        outputs = model(patches_left[levels[i]].to(device)) 
                        loss += criterion(outputs[:], label[:, i].to(device))  
                        outputs = model(patches_right[levels[i]].to(device)) 
                        loss += criterion(outputs[:], label[:, 5+i].to(device))                
                    epoch_len = len(train_data) // train_loader.batch_size
                    total_loss = loss.item() / (2*C)
                    metric_eval += total_loss
                except Exception as e:
                    print(f"Exception {e}")
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
    levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    model.eval()
    num_correct = 0
    metric_count = 0
    total_loss = 0
    y_true = []
    y_pred = []

    C=5

    for batch_data in tqdm(test_loader):
        
        patches_left, patches_right, label, study_id = batch_data
            
        with torch.no_grad():
            loss = 0
            for i in range(C):    
                # outputs, _ = model(patches[levels[i]].to(device))
                outputs = model(patches_left[levels[i]].to(device))
                
                loss += criterion(outputs[:], label[:, i].to(device))                

                # print(list(outputs.argmax(dim=-1).cpu().numpy()))
                y_pred += list(outputs.cpu().numpy())
                y_true += list(label[:,i].cpu().numpy())
                value = torch.eq(outputs.argmax(dim=-1).cpu(), label[:, i])
                metric_count += len(value)
                num_correct += value.sum().item()
                
                outputs = model(patches_right[levels[i]].to(device))
                
                loss += criterion(outputs[:], label[:, C+i].to(device))                

                # print(list(outputs.argmax(dim=-1).cpu().numpy()))
                y_pred += list(outputs.cpu().numpy())
                y_true += list(label[:,i].cpu().numpy())
                value = torch.eq(outputs.argmax(dim=-1).cpu(), label[:, i])
                metric_count += len(value)
                num_correct += value.sum().item()
                      
                # print(outputs)
            total_loss += loss.item() / (2*C)

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
        "sag-T1": ["left_neural_foraminal_narrowing", 
                   "right_neural_foraminal_narrowing"],
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


    # subjects to exclude

    # exclude = list(np.load(exclude_file))

    # Build dataset and dataloader
    coordinates = pd.read_csv("./data/train_label_coordinates.csv")
    description = pd.read_csv("./data/train_series_descriptions.csv")
    
    
    transform = Compose(
        [
            NormalizeIntensityd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]),
            ResizeWithPadOrCropd(keys=["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"],
                                spatial_size=(16, 64, 64)),
        ]    
    )

    exclude = ["3637444890"]

    root_dir="../TotalSpineSeg"

    vol_paths = glob.glob(root_dir + "/data/sub*T2w.nii.gz")
    seg_paths = glob.glob(root_dir+"/output/step1_output/*T2w.nii.gz")

    for x in exclude:
        for pth in vol_paths:
            if x in pth:
                vol_paths.remove(pth)
                print(x)
        for pth in seg_paths:
            if x in pth:
                seg_paths.remove(pth) 
                print(x)

    vol_paths.sort()
    seg_paths.sort()

    train_vols, val_vols, train_seg, val_seg = train_test_split(vol_paths, seg_paths, test_size=0.5, random_state=42)
    val_vols, test_vols, val_seg, test_seg = train_test_split(val_vols, val_seg, test_size=0.5, random_state=42)

    train_df = pd.read_csv("./data/train.csv")
    train_df = train_df.dropna()
    text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe":2}
    train_df = train_df.replace(text2int)


    train_df = train_df[["study_id",
                        "left_neural_foraminal_narrowing_l1_l2",
                        "left_neural_foraminal_narrowing_l2_l3",
                        "left_neural_foraminal_narrowing_l3_l4",
                        "left_neural_foraminal_narrowing_l4_l5",
                        "left_neural_foraminal_narrowing_l5_s1",
                        "right_neural_foraminal_narrowing_l1_l2",
                        "right_neural_foraminal_narrowing_l2_l3",
                        "right_neural_foraminal_narrowing_l3_l4",
                        "right_neural_foraminal_narrowing_l4_l5",
                        "right_neural_foraminal_narrowing_l5_s1"]]

    df = pd.DataFrame(train_df[["left_neural_foraminal_narrowing_l1_l2",
                                "left_neural_foraminal_narrowing_l2_l3",
                                "left_neural_foraminal_narrowing_l3_l4",
                                "left_neural_foraminal_narrowing_l4_l5",
                                "left_neural_foraminal_narrowing_l5_s1",
                                "right_neural_foraminal_narrowing_l1_l2",
                                "right_neural_foraminal_narrowing_l2_l3",
                                "right_neural_foraminal_narrowing_l3_l4",
                                "right_neural_foraminal_narrowing_l4_l5",
                                "right_neural_foraminal_narrowing_l5_s1"]].sum(axis=1))

    df.rename(columns={0: "count"}, inplace=True)
    idx = np.where(df.values[:, 0]<=5)
    exclude = np.random.choice(idx[0], size=300, replace=False)
    exclude = list(train_df.iloc[exclude]["study_id"].values)


    train_data = ForaminalNarrowingDataset(root_dir=root_dir, 
                                           vol_paths=train_vols, 
                                           seg_paths=train_seg, 
                                           transform=transform,
                                           exclude = exclude)
    val_data = ForaminalNarrowingDataset(root_dir=root_dir, 
                                         vol_paths=val_vols, 
                                         seg_paths=val_seg, 
                                         transform=transform,
                                         exclude = exclude)
    test_data = ForaminalNarrowingDataset(root_dir=root_dir, 
                                          vol_paths=test_vols, 
                                          seg_paths=test_seg, 
                                          transform=transform,
                                          exclude = list())
    
    print("Train dataset length :", len(train_data))
    print("Val dataset length :", len(val_data))
    print("Test dataset length :", len(test_data))

    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True)

    writer_train = SummaryWriter("./runs/experiment_ ")

    device = torch.device(f"{gpu}" if torch.cuda.is_available() else "cpu")

    print("device =", device)
    
    autocast = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.half)
    scaler = torch.cuda.amp.GradScaler(enabled=scaler, init_scale=4096)


    # ResNet17
    model = ResNet(
                    block="basic",
                    layers=[2, 2, 2, 2],
                    block_inplanes=[64, 128, 256, 512],
                    spatial_dims=3,
                    n_input_channels=1,
                    num_classes=3,
                ).to(device)
    
    # ViT
    
    # model = ViT(in_channels=1, 
    #             img_size=(4, 64, 64), 
    #             patch_size=(2, 32, 32),
    #             spatial_dims=3,
    #             pos_embed='conv', 
    #             post_activation = None,
    #             classification=True,
    #             num_classes=3
    #             ).to(device)
    
    # ResNet34
    
    # model = ResNet(
    #                 block="basic",
    #                 layers=[3, 4, 6, 3],
    #                 block_inplanes=[64, 128, 256, 512],
    #                 spatial_dims=3,
    #                 n_input_channels=1,
    #                 num_classes=3,
    #             ).to(device)

    # Models, optimization method, loss criterion
    
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1., 2., 4.]).to(device), reduction="mean")
    # criterions.append(torch.nn.CrossEntropyLoss(weight=torch.Tensor(weigth).to(device)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

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

    # model = ResNet(
    #         block="basic",
    #         layers=[3, 4, 6, 3],
    #         block_inplanes=[64, 128, 256, 512],
    #         spatial_dims=3,
    #         n_input_channels=1,
    #         num_classes=3,
    # ).to(device)
    
    # model = ViT(in_channels=1, 
    #                 img_size=(4, 64, 64), 
    #                 patch_size=(2, 32, 32),
    #                 spatial_dims=2,
    #                 pos_embed='conv', 
    #                 post_activation = None,
    #                 classification=True,
    #                 num_classes=3
    #                 ).to(device)
    model = ResNet(
                block="basic",
                layers=[2, 2, 2, 2],
                block_inplanes=[64, 128, 256, 512],
                spatial_dims=3,
                n_input_channels=1,
                num_classes=3,
            ).to(device)

    
    
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
