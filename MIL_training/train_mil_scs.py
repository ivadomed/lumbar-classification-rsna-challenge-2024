''' train the SCS MIL model '''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, ConcatDataset
from training_utils import CosineAnnealingStabilizeLR, weight_challenge, train_epoch, validate, visualize_batch
import wandb
from tqdm import tqdm
from prepare_scs import prepare_data_scs
from mil_definition import MILmodel
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import math
import timm 




def train_model_scs(
    encoder,
    data_dir,
    csv_file,
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    encoder_lr=1e-5,  # Learning rate spécifique pour le ConvNext
    freeze_encoder_epoch=5,  # Époque à partir de laquelle on freeze le ConvNext
    encoder_cosine_epochs=3,  # Nombre d'époques pour atteindre le minimum du cosine pour l'encoder
    other_cosine_epochs=6,  # Nombre d'époques pour atteindre le minimum du cosine pour le reste
    eta_min_factor_encoder=0.04,  # Facteur pour calculer eta_min de l'encoder (par rapport à encoder_lr)
    eta_min_factor_other=0.04,  # Facteur pour calculer eta_min du reste (par rapport à learning_rate)
    num_layers=1,
    device='cuda',
):
    # Initialize wandb
    wandb.init(
        project="lumbar-mil-scs",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "encoder_lr": encoder_lr,
            "freeze_encoder_epoch": freeze_encoder_epoch,
            "encoder_cosine_epochs": encoder_cosine_epochs,
            "other_cosine_epochs": other_cosine_epochs,
            "eta_min_factor_encoder": eta_min_factor_encoder,
            "eta_min_factor_other": eta_min_factor_other,
            "scheduler": "CosineAnnealing",
            "architecture": "ConvNeXt-Small-MIL",
            "num_layers": num_layers
        }
    )

    # create a folder with a random name in the current directory
    folder_name = f"mil_model_scs"
    os.makedirs(folder_name, exist_ok=True)

    # Prepare data
    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    # Create datasets
    train_data = prepare_data_scs(train_dir, csv_file, random=True)
    val_data = prepare_data_scs(val_dir, csv_file, random=False)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # Initialize model
    model = MILmodel(encoder=encoder, num_layers=num_layers).to(device)

    # Loss function - CrossEntropyLoss with class weights if needed
    criterion = nn.CrossEntropyLoss(weight=weight_challenge)

    # Séparer les paramètres du ConvNext et du reste du modèle
    encoder_params = model.encoder.parameters()
    other_params = [p for n, p in model.named_parameters() if not n.startswith('encoder')]

    # Optimizer avec learning rates différents
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': other_params, 'lr': learning_rate}
    ], weight_decay=0.01)

    # Learning rate schedulers séparés avec périodes différentes
    encoder_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * encoder_cosine_epochs,  # Période spécifique pour l'encoder
        eta_min=encoder_lr * eta_min_factor_encoder,  # Minimum learning rate pour l'encoder
    )
    
    other_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * other_cosine_epochs,  # Période spécifique pour le reste
        eta_min=learning_rate * eta_min_factor_other,  # Minimum learning rate pour le reste
    )

    # Log initial learning rates and minimum values
    wandb.log({
        "initial_encoder_lr": encoder_lr,
        "initial_other_lr": learning_rate,
        "min_encoder_lr": encoder_lr * eta_min_factor_encoder,
        "min_other_lr": learning_rate * eta_min_factor_other,
        "encoder_cosine_epochs": encoder_cosine_epochs,
        "other_cosine_epochs": other_cosine_epochs,
        "eta_min_factor_encoder": eta_min_factor_encoder,
        "eta_min_factor_other": eta_min_factor_other
    })

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Freeze le ConvNext après freeze_encoder_epoch époques
        if epoch >= freeze_encoder_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("ConvNext encoder frozen")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            (encoder_scheduler, other_scheduler), device,
            epoch=epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "encoder_learning_rate": encoder_scheduler.get_last_lr()[0],
            "other_learning_rate": other_scheduler.get_last_lr()[0],
            "encoder_frozen": epoch >= freeze_encoder_epoch,
            "encoder_lr_percentage": (encoder_scheduler.get_last_lr()[0] / encoder_lr) * 100,  # Pourcentage du LR initial
            "other_lr_percentage": (other_scheduler.get_last_lr()[0] / learning_rate) * 100  # Pourcentage du LR initial
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'encoder_scheduler_state_dict': encoder_scheduler.state_dict(),
                'other_scheduler_state_dict': other_scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(folder_name, f"best_mil_model.pth"))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # adds a json file in the folder with the config and the best loss
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(dict(wandb.config), f)
    with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
        json.dump({'best_loss': best_val_loss}, f)

    wandb.finish()
    return model


# launch a training with the SCS model
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    data_dir = '../../duke/public/rsna_challenge/20250212nii_data_splits'
    csv_file = '../../duke/public/rsna_challenge/dcom_data/train.csv'


    convnext_small = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=True, num_classes=0)


    # Train model
    model = train_model_scs(
        convnext_small,
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=16,
        batch_size=2,
        learning_rate=5e-5,
        encoder_lr=5e-5,  # Learning rate plus faible pour le ConvNext
        freeze_encoder_epoch=20,  # Freeze le ConvNext après 3 époques
        encoder_cosine_epochs=16,  # Le ConvNext atteint son minimum en 2 époques
        other_cosine_epochs=16,  # Le reste du modèle atteint son minimum en 4 époques
        eta_min_factor_encoder=0.05,  # Le lr de l'encoder descend à 4% de sa valeur initiale
        eta_min_factor_other=0.05,  # Le lr du reste descend à 4% de sa valeur initiale
        num_layers=2,
        device=device,  
    )

