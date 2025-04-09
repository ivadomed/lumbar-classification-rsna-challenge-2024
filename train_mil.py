import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from tqdm import tqdm
from prepare_data_mil import prepare_data_scs, prepare_data_sas, prepare_data_nfn
from mil_definition import MILmodel, convnext_small
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import math


# use the challenge's loss function : weighted cross entropy
# with weights 1, 2, 4 for the 3 classes
weight_challenge = torch.tensor([1.0, 2.0, 4.0]).cuda()


class CosineAnnealingStabilizeLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingStabilizeLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


# Function to train the model for one epoch
def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    schedulers,
    device,
    aux_weight,
    epoch=None  # Add epoch parameter
):
    model.train()
    running_loss = 0.0
    correct_main = 0
    correct_aux = 0
    total = 0

    encoder_scheduler, other_scheduler = schedulers

    pbar = tqdm(train_loader, desc='Training')
    for i, batch in enumerate(pbar):
        # Get data
        bags = batch['bag'].to(device)  # Shape: [B, 6, 1, 384, 384]
        labels = batch['label'].to(device)  # Shape: [B]

        # Visualize first batch of each epoch (after moving to device)
        if i == 0 and epoch is not None:
            visualize_batch(batch, epoch)
            
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        main_output, aux_output = model(bags)

        # Calculate losses
        main_loss = criterion(main_output, labels)
        aux_loss = criterion(aux_output, labels)
        # Auxiliary loss weighted by aux_weight
        loss = main_loss + aux_weight * aux_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rates
        encoder_scheduler.step()
        other_scheduler.step()

        # Calculate accuracy
        _, predicted_main = torch.max(main_output, 1)
        _, predicted_aux = torch.max(aux_output, 1)
        total += labels.size(0)
        correct_main += (predicted_main == labels).sum().item()
        correct_aux += (predicted_aux == labels).sum().item()

        # Update statistics
        running_loss += loss.item()

        # Update progress bar with learning rates
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'main_acc': f'{100 * correct_main / total:.2f}%',
            'aux_acc': f'{100 * correct_aux / total:.2f}%',
            'enc_lr': f'{encoder_scheduler.get_last_lr()[0]:.2e}',
            'oth_lr': f'{other_scheduler.get_last_lr()[0]:.2e}'
        })

    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    main_acc = 100 * correct_main / total
    aux_acc = 100 * correct_aux / total

    return epoch_loss, main_acc, aux_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, aux_weight):
    model.eval()
    running_loss = 0.0
    correct_main = 0
    correct_aux = 0
    total = 0

    for batch in tqdm(val_loader, desc='Validation'):
        # Get data
        bags = batch['bag'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        main_output, aux_output = model(bags)

        # Calculate losses
        main_loss = criterion(main_output, labels)
        aux_loss = criterion(aux_output, labels)
        loss = main_loss + aux_weight * aux_loss

        # Calculate accuracy
        _, predicted_main = torch.max(main_output, 1)
        _, predicted_aux = torch.max(aux_output, 1)
        total += labels.size(0)
        correct_main += (predicted_main == labels).sum().item()
        correct_aux += (predicted_aux == labels).sum().item()

        running_loss += loss.item()

    # Calculate epoch statistics
    val_loss = running_loss / len(val_loader)
    main_acc = 100 * correct_main / total
    aux_acc = 100 * correct_aux / total

    return val_loss, main_acc, aux_acc


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


def train_model_scs(
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
    aux_loss_weight=0,
    aux_loss_schedule='constant',
    num_layers=1,
    device='cuda'
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
            "aux_loss_weight": aux_loss_weight,
            "aux_loss_schedule": aux_loss_schedule,
            "num_layers": num_layers
        }
    )

    # create a folder with a random name in the current directory
    folder_name = f"mil_model_scs{random.randint(0, 1000000)}"
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
    model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)

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
        eta_min=encoder_lr * eta_min_factor_encoder  # Minimum learning rate pour l'encoder
    )
    
    other_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * other_cosine_epochs,  # Période spécifique pour le reste
        eta_min=learning_rate * eta_min_factor_other  # Minimum learning rate pour le reste
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

    def get_aux_weight(epoch):
        if aux_loss_schedule == 'constant':
            return aux_loss_weight
        elif aux_loss_schedule == 'linear_decay':
            return aux_loss_weight * (1 - epoch / num_epochs)
        elif aux_loss_schedule == 'cosine_decay':
            return aux_loss_weight * np.cos(np.pi * epoch / (2 * num_epochs))
        else:
            raise ValueError(f"Unknown schedule: {aux_loss_schedule}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Freeze le ConvNext après freeze_encoder_epoch époques
        if epoch >= freeze_encoder_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("ConvNext encoder frozen")

        # Get current auxiliary loss weight
        current_aux_weight = get_aux_weight(epoch)

        # Train
        train_loss, train_main_acc, train_aux_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            (encoder_scheduler, other_scheduler), device,
            aux_weight=current_aux_weight,
            epoch=epoch
        )

        # Validate
        val_loss, val_main_acc, val_aux_acc = validate(
            model, val_loader, criterion, device,
            aux_weight=current_aux_weight
        )

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_main_acc": train_main_acc,
            "train_aux_acc": train_aux_acc,
            "val_loss": val_loss,
            "val_main_acc": val_main_acc,
            "val_aux_acc": val_aux_acc,
            "encoder_learning_rate": encoder_scheduler.get_last_lr()[0],
            "other_learning_rate": other_scheduler.get_last_lr()[0],
            "aux_loss_weight": current_aux_weight,
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
                'val_main_acc': val_main_acc,
                'val_aux_acc': val_aux_acc,
            }, os.path.join(folder_name, f"best_mil_model.pth"))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # adds a json file in the folder with the config and the best loss
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(dict(wandb.config), f)
    with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
        json.dump({'best_loss': best_val_loss}, f)

    wandb.finish()
    return model

def train_model_sas(
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
    aux_loss_weight=0,
    aux_loss_schedule='constant',
    num_layers=1,
    device='cuda'
):
    # Initialize wandb
    wandb.init(
        project="lumbar-mil-sas",
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
            "aux_loss_weight": aux_loss_weight,
            "aux_loss_schedule": aux_loss_schedule,
            "num_layers": num_layers
        }
    )

    # create a folder with a random name in the current directory
    folder_name = f"mil_model_sas{random.randint(0, 1000000)}"
    os.makedirs(folder_name, exist_ok=True)

    # Prepare data
    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    # Create datasets
    train_data_left, train_data_right = prepare_data_sas(train_dir, csv_file, random=True)
    val_data_left, val_data_right = prepare_data_sas(val_dir, csv_file, random=False)
    train_data = ConcatDataset([train_data_left, train_data_right])
    val_data = ConcatDataset([val_data_left, val_data_right])

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # Initialize model
    model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)

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
        eta_min=encoder_lr * eta_min_factor_encoder  # Minimum learning rate pour l'encoder
    )
    
    other_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * other_cosine_epochs,  # Période spécifique pour le reste
        eta_min=learning_rate * eta_min_factor_other  # Minimum learning rate pour le reste
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

    def get_aux_weight(epoch):
        if aux_loss_schedule == 'constant':
            return aux_loss_weight
        elif aux_loss_schedule == 'linear_decay':
            return aux_loss_weight * (1 - epoch / num_epochs)
        elif aux_loss_schedule == 'cosine_decay':
            return aux_loss_weight * np.cos(np.pi * epoch / (2 * num_epochs))
        else:
            raise ValueError(f"Unknown schedule: {aux_loss_schedule}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Freeze le ConvNext après freeze_encoder_epoch époques
        if epoch >= freeze_encoder_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("ConvNext encoder frozen")

        # Get current auxiliary loss weight
        current_aux_weight = get_aux_weight(epoch)

        # Train
        train_loss, train_main_acc, train_aux_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            (encoder_scheduler, other_scheduler), device,
            aux_weight=current_aux_weight,
            epoch=epoch
        )

        # Validate
        val_loss, val_main_acc, val_aux_acc = validate(
            model, val_loader, criterion, device,
            aux_weight=current_aux_weight
        )

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_main_acc": train_main_acc,
            "train_aux_acc": train_aux_acc,
            "val_loss": val_loss,
            "val_main_acc": val_main_acc,
            "val_aux_acc": val_aux_acc,
            "encoder_learning_rate": encoder_scheduler.get_last_lr()[0],
            "other_learning_rate": other_scheduler.get_last_lr()[0],
            "aux_loss_weight": current_aux_weight,
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
                'val_main_acc': val_main_acc,
                'val_aux_acc': val_aux_acc,
            }, os.path.join(folder_name, f"best_mil_model.pth"))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # adds a json file in the folder with the config and the best loss
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(dict(wandb.config), f)
    with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
        json.dump({'best_loss': best_val_loss}, f)

    wandb.finish()
    return model


def train_model_nfn(
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
    aux_loss_weight=0,
    aux_loss_schedule='constant',
    num_layers=1,
    device='cuda'
):
    # Initialize wandb
    wandb.init(
        project="lumbar-mil-nfn",
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
            "aux_loss_weight": aux_loss_weight,
            "aux_loss_schedule": aux_loss_schedule,
            "num_layers": num_layers
        }
    )

    # create a folder with a random name in the current directory
    folder_name = f"mil_model_nfn{random.randint(0, 1000000)}"
    os.makedirs(folder_name, exist_ok=True)

    # Prepare data
    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    # Create datasets
    train_data = prepare_data_nfn(train_dir, csv_file, random=True)
    val_data= prepare_data_nfn(val_dir, csv_file, random=False)
    # train_data = ConcatDataset([train_data_left, train_data_right])
    # val_data = ConcatDataset([val_data_left, val_data_right])

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Initialize model
    model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)

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
        eta_min=encoder_lr * eta_min_factor_encoder  # Minimum learning rate pour l'encoder
    )
    
    other_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * other_cosine_epochs,  # Période spécifique pour le reste
        eta_min=learning_rate * eta_min_factor_other  # Minimum learning rate pour le reste
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

    def get_aux_weight(epoch):
        if aux_loss_schedule == 'constant':
            return aux_loss_weight
        elif aux_loss_schedule == 'linear_decay':
            return aux_loss_weight * (1 - epoch / num_epochs)
        elif aux_loss_schedule == 'cosine_decay':
            return aux_loss_weight * np.cos(np.pi * epoch / (2 * num_epochs))
        else:
            raise ValueError(f"Unknown schedule: {aux_loss_schedule}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Freeze le ConvNext après freeze_encoder_epoch époques
        if epoch >= freeze_encoder_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("ConvNext encoder frozen")

        # Get current auxiliary loss weight
        current_aux_weight = get_aux_weight(epoch)

        # Train
        train_loss, train_main_acc, train_aux_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            (encoder_scheduler, other_scheduler), device,
            aux_weight=current_aux_weight,
            epoch=epoch
        )

        # Validate
        val_loss, val_main_acc, val_aux_acc = validate(
            model, val_loader, criterion, device,
            aux_weight=current_aux_weight
        )

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_main_acc": train_main_acc,
            "train_aux_acc": train_aux_acc,
            "val_loss": val_loss,
            "val_main_acc": val_main_acc,
            "val_aux_acc": val_aux_acc,
            "encoder_learning_rate": encoder_scheduler.get_last_lr()[0],
            "other_learning_rate": other_scheduler.get_last_lr()[0],
            "aux_loss_weight": current_aux_weight,
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
                'val_main_acc': val_main_acc,
                'val_aux_acc': val_aux_acc,
            }, os.path.join(folder_name, f"best_mil_model.pth"))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # adds a json file in the folder with the config and the best loss
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(dict(wandb.config), f)
    with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
        json.dump({'best_loss': best_val_loss}, f)

    wandb.finish()
    return model


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    data_dir = '../../duke/public/rsna_challenge/20250408nii_data'
    csv_file = '../../duke/public/rsna_challenge/dcom_data/train.csv'

    # Train model
    model = train_model_nfn(
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=20,
        batch_size=2,
        learning_rate=5e-5,
        encoder_lr=5e-5,  # Learning rate plus faible pour le ConvNext
        freeze_encoder_epoch=3,  # Freeze le ConvNext après 3 époques
        encoder_cosine_epochs=10,  # Le ConvNext atteint son minimum en 2 époques
        other_cosine_epochs=6,  # Le reste du modèle atteint son minimum en 4 époques
        eta_min_factor_encoder=0.1,  # Le lr de l'encoder descend à 4% de sa valeur initiale
        eta_min_factor_other=0.1,  # Le lr du reste descend à 4% de sa valeur initiale
        aux_loss_weight=0,
        aux_loss_schedule='constant',
        num_layers=3,
        device=device
    )
