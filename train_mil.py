import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from prepare_data_mil import prepare_data, get_transforms
from mil_definition import MILmodel, convnext_small
import numpy as np
import matplotlib.pyplot as plt
import random
import json


# use the challenge's loss function : weighted cross entropy
# with weights 1, 2, 4 for the 3 classes
weight_challenge = torch.tensor([1.0, 2.0, 4.0]).cuda()


# Function to train the model for one epoch
def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    aux_weight,
    epoch=None  # Add epoch parameter
):
    model.train()
    running_loss = 0.0
    correct_main = 0
    correct_aux = 0
    total = 0

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

        # Update learning rate
        scheduler.step()

        # Calculate accuracy
        _, predicted_main = torch.max(main_output, 1)
        _, predicted_aux = torch.max(aux_output, 1)
        total += labels.size(0)
        correct_main += (predicted_main == labels).sum().item()
        correct_aux += (predicted_aux == labels).sum().item()

        # Update statistics
        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'main_acc': f'{100 * correct_main / total:.2f}%',
            'aux_acc': f'{100 * correct_aux / total:.2f}%'
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


def train_model(
    data_dir,
    csv_file,
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    aux_loss_weight=0,
    aux_loss_schedule='constant',
    num_layers=1,
    device='cuda'
):
    # Initialize wandb
    wandb.init(
        project="lumbar-mil",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "scheduler": "CosineAnnealing",
            "architecture": "ConvNeXt-Small-MIL",
            "aux_loss_weight": aux_loss_weight,
            "aux_loss_schedule": aux_loss_schedule,
            "num_layers": num_layers
        }
    )

    # create a folder with a random name in the current directory
    folder_name = f"mil_model_{random.randint(0, 1000000)}"
    os.makedirs(folder_name, exist_ok=True)

    # Prepare data
    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    # Create datasets
    train_data = prepare_data(train_dir, csv_file, random=True)
    val_data = prepare_data(val_dir, csv_file, random=False)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # Initialize model
    model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)

    # Loss function - CrossEntropyLoss with class weights if needed
    criterion = nn.CrossEntropyLoss(weight=weight_challenge)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                            weight_decay=0.01)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * 6,  # Total number of steps
        eta_min=8e-7  # Minimum learning rate
    )

    def get_aux_weight(epoch):
        if aux_loss_schedule == 'constant':
            return aux_loss_weight
        elif aux_loss_schedule == 'linear_decay':
            # Décroissance linéaire de aux_loss_weight à 0
            return aux_loss_weight * (1 - epoch / num_epochs)
        elif aux_loss_schedule == 'cosine_decay':
            # Décroissance en cosinus de aux_loss_weight à 0
            return aux_loss_weight * np.cos(np.pi * epoch / (2 * num_epochs))
        else:
            raise ValueError(f"Unknown schedule: {aux_loss_schedule}")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Get current auxiliary loss weight
        current_aux_weight = get_aux_weight(epoch)

        # Train
        train_loss, train_main_acc, train_aux_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            aux_weight=current_aux_weight,
            epoch=epoch  # Pass the epoch number
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
            "learning_rate": scheduler.get_last_lr()[0],
            "aux_loss_weight": current_aux_weight
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the model in a folder with rando
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_main_acc': val_main_acc,
                'val_aux_acc': val_aux_acc,
            }, os.path.join(folder_name, f"best_mil_model.pth"))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

    # adds a json file in the folder with the config and the best loss
    with open(os.path.join(folder_name, 'config.json'), 'w') as f:
        json.dump(wandb.config, f)
    with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
        json.dump({'best_loss': best_val_loss}, f)

    wandb.finish()
    return model


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    data_dir = '/home/ge.polymtl.ca/p121315/duke/public/rsna_challenge/20250212nii_data_splits'  # Update with your data path
    csv_file = '/home/ge.polymtl.ca/p121315/duke/public/rsna_challenge/dcom_data/train.csv'  # Update with your CSV path

    # Train model
    model = train_model(
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=10,
        batch_size=8,
        learning_rate=5e-5,
        aux_loss_weight=0,
        aux_loss_schedule='constant',
        num_layers=1,  # 0 pour désactiver le RNN
        device=device
    )
