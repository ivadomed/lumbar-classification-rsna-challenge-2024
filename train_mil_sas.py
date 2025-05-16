''' train the SAS MIL model '''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, ConcatDataset
from training_utils import CosineAnnealingStabilizeLR, weight_challenge, train_epoch, validate
import wandb
from tqdm import tqdm
from prepare_sas import prepare_data_sas 
from mil_definition import MILmodel
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import math
import timm 

# function to visualize a batch of images and save them to wandb for sas
def visualize_batch(batch, epoch):
    """
    Visualize a batch of images and save them to wandb
    batch: dictionary containing 'bag' tensor of shape [B, 6, 1, 384, 384] and 'label'
    epoch: current epoch number
    """
    try:
        # Get the first batch and ensure it's on CPU
        images = batch['bag'].cpu().detach()  # Shape: [B, 6, 1, 384, 384]
        labels = batch['label'].cpu().detach()
        
        # Take only the first 4 samples to avoid too large figures
        n_samples = min(4, images.shape[0])
        
        # Create a figure with subplots for each sample and its 6 slices
        fig, axes = plt.subplots(n_samples, 6, figsize=(20, 4*n_samples))
        if n_samples == 1:
            axes = axes[None, :]  # Add dimension for consistent indexing
        
        for i in range(n_samples):
            for j in range(6):
                # Get the image slice and ensure it's a valid image
                img = images[i, j, 0].numpy()
                
                # Normalize the image for better visualization
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                # Plot the image
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                
                # Add title only to the first row
                if i == 0:
                    axes[i, j].set_title(f'Slice {j+1}')
            
            # Add label information on the left
            axes[i, 0].set_ylabel(f'Sample {i+1}\nLabel: {labels[i].item()}')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"batch_visualization_epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not visualize batch: {str(e)}")
        plt.close('all')  # Ensure all figures are closed in case of error

# main function to train the SAS MIL model
def train_model_sas(
    encoder,
    data_dir,
    csv_file,
    num_epochs=20,
    batch_size=8,
    learning_rate=6e-5,
    freeze_encoder_epoch=6,
    cosine_epochs=10,
    eta_min_factor=0.04,
    fine_tune_learning_rate=2e-5,
    fine_tune_cosine_epochs=5,
    fine_tune_eta_min_factor=0.1,
    num_layers=1,
    device='cuda',
    save_4_wv=False,
    pretrained_model_path=None
):
    """
    Train a single model with a two-phase approach:
    1. Initial training with unweighted loss and single learning rate
    2. Fine-tuning with frozen encoder and weighted loss
    
    Args:
        pretrained_model_path (str, optional): Path to a pretrained model to load and fine-tune.
            If None, a new model will be initialized.
    """
    # Initialize wandb
    wandb.init(
        project="lumbar-mil-sas",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "freeze_encoder_epoch": freeze_encoder_epoch,
            "cosine_epochs": cosine_epochs,
            "eta_min_factor": eta_min_factor,
            "fine_tune_learning_rate": fine_tune_learning_rate,
            "fine_tune_cosine_epochs": fine_tune_cosine_epochs,
            "fine_tune_eta_min_factor": fine_tune_eta_min_factor,
            "scheduler": "CosineAnnealing",
            "architecture": "ConvNeXt-Small-MIL",
            "num_layers": num_layers,
            "pretrained_model_path": pretrained_model_path
        }
    )

    # Create a folder for the model
    folder_name = f"mil_model_sas_{random.randint(0, 1000000)}"
    os.makedirs(folder_name, exist_ok=True)

    # Prepare data
    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    save_4_wv = save_4_wv

    # Create datasets
    train_data_left, train_data_right = prepare_data_sas(train_dir, csv_file, random=True)
    val_data_left, val_data_right = prepare_data_sas(val_dir, csv_file, random=False)
    train_data = ConcatDataset([train_data_left, train_data_right])
    val_data = ConcatDataset([val_data_left, val_data_right])

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    '''severe_right, severe_left = prepare_data_sas_option(val_dir, csv_file, option=2, random=False)
    severe_loader = DataLoader(ConcatDataset([severe_right, severe_left]), batch_size=2, shuffle=False, num_workers=8)

    moderate_right, moderate_left = prepare_data_sas_option(val_dir, csv_file, option=1, random=False)
    moderate_loader = DataLoader(ConcatDataset([moderate_right, moderate_left]), batch_size=2, shuffle=False, num_workers=8)
    '''
    # Loss functions
    unweighted_criterion = nn.CrossEntropyLoss()  # For phase 1 training and validation
    weighted_criterion = nn.CrossEntropyLoss(weight=weight_challenge)  # For phase 2 training and validation


    # Initialize model
    if pretrained_model_path is not None:
        '''print(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path)
        model = MILmodel(encoder=encoder, num_layers=num_layers).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded successfully")
        in_cv, in_mean_loss, in_var_loss = run_inference_on_validation_set(model, severe_loader, device, weighted_criterion)
        print(f"Initial CV for severe cases: {in_cv}, mean loss: {in_mean_loss}, var loss: {in_var_loss}")
    '''
    else:
        model = MILmodel(encoder=encoder, num_layers=num_layers).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate schedulers
    initial_scheduler = CosineAnnealingStabilizeLR(
        optimizer,
        T_max=len(train_loader) * cosine_epochs,
        eta_min=learning_rate * eta_min_factor,
    )

    # Fine-tuning scheduler will be initialized when needed
    fine_tune_scheduler = None

    # Initialize lists to store losses for CV calculation
    train_losses = []
    val_losses = []
    val_weighted_losses = []

    # Training loop
    best_val_loss = float('inf')
    best_val_weighted_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Switch to phase 2 if needed
        if epoch >= freeze_encoder_epoch:
            if fine_tune_scheduler is None:
                # Initialize fine-tuning scheduler
                fine_tune_scheduler = CosineAnnealingStabilizeLR(
                    optimizer,
                    T_max=len(train_loader) * fine_tune_cosine_epochs,
                    eta_min=fine_tune_learning_rate * fine_tune_eta_min_factor,
                )
                # Set the new learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = fine_tune_learning_rate
                print(f"Initialized fine-tuning scheduler with learning rate {fine_tune_learning_rate}")
            
            # Freeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("ConvNext encoder frozen - Switching to weighted loss")
            # Switch to weighted criterion for training
            criterion = weighted_criterion
            # Use fine-tuning scheduler
            scheduler = fine_tune_scheduler
        else:
            criterion = weighted_criterion
            scheduler = initial_scheduler

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            (scheduler, scheduler), device,
            epoch=epoch
        )

        # Validate with both criteria
        val_loss, val_acc = validate(
            model, val_loader, unweighted_criterion, device
        )
        val_weighted_loss, val_weighted_acc = validate(
            model, val_loader, weighted_criterion, device
        )

        # Store losses for CV calculation
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_weighted_losses.append(val_weighted_loss)

        # Calculate coefficient of variation
        '''val_severe_cv, val_severe_mean_loss, val_severe_var_loss = run_inference_on_validation_set(model, severe_loader, device, weighted_criterion)
        print(f"CV for severe cases: {val_severe_cv}, mean loss: {val_severe_mean_loss}, var loss: {val_severe_var_loss}")
        val_moderate_cv, val_moderate_mean_loss, val_moderate_var_loss = run_inference_on_validation_set(model, moderate_loader, device, weighted_criterion)
        '''

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_weighted_loss": val_weighted_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        # Save best model
        if val_weighted_loss < best_val_weighted_loss:
            best_val_weighted_loss = val_weighted_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'initial_scheduler_state_dict': initial_scheduler.state_dict(),
                'fine_tune_scheduler_state_dict': fine_tune_scheduler.state_dict() if fine_tune_scheduler else None,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(folder_name, f"best_mil_model.pth"))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # adds a json file in the folder with the config and the best loss
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(dict(wandb.config), f)
    with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
        json.dump({'best_loss': best_val_weighted_loss}, f)

    wandb.finish()
    return model

# lauching a training with the SAS model
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    data_dir = '../../duke/public/rsna_challenge/20250212nii_data_splits'
    #data_dir = '../../duke/public/rsna_challenge/20250410nii_folds''
    csv_file = '../../duke/public/rsna_challenge/dcom_data/train.csv'


    convnext_small = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=True, num_classes=0)


    # Train model
    model = train_model_sas(
        convnext_small,
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=16,
        batch_size=2,
        learning_rate=5e-5,  # Learning rate plus faible pour le ConvNext
        freeze_encoder_epoch=20,  # Freeze le ConvNext après 3 époques
        cosine_epochs=12,  # Le ConvNext atteint son minimum en 2 époques
        eta_min_factor=0.05,  # Le lr de l'encoder descend à 4% de sa valeur initiale
        num_layers=2,
        device=device
    )


