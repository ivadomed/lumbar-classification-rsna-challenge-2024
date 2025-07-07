''' utility functions for training the models '''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import wandb


# use the challenge's loss function : weighted cross entropy
# with weights 1, 2, 4 for the 3 classes
weight_challenge = torch.tensor([1.0, 2.0, 4.0]).cuda()

# custom learning rate scheduler
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

# function to plot the first batch of each epoch on wandb
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


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizers,          # Tuple: (encoder_optimizer, other_optimizer)
    schedulers,          # Tuple: (encoder_scheduler, other_scheduler)
    device,
    epoch=None
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    encoder_optimizer, other_optimizer = optimizers
    encoder_scheduler, other_scheduler = schedulers

    pbar = tqdm(train_loader, desc='Training')
    for i, batch in enumerate(pbar):
        bags = batch['bag'].to(device)
        labels = batch['label'].to(device)

        if i == 0 and epoch is not None:
            visualize_batch({k: v.cpu() for k, v in batch.items()}, epoch)

        encoder_optimizer.zero_grad()
        other_optimizer.zero_grad()

        main_output = model(bags)
        loss = criterion(main_output, labels)

        loss.backward()

        encoder_optimizer.step()
        other_optimizer.step()

        _, predicted = torch.max(main_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    encoder_scheduler.step()
    other_scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    acc = 100 * correct / total

    return epoch_loss, acc



# Function to validate the model
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(val_loader, desc='Validation'):
        # Get data
        bags = batch['bag'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        main_output = model(bags)

        # Calculate losses
        loss = criterion(main_output, labels)

        # Calculate accuracy
        _, predicted = torch.max(main_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    # Calculate epoch statistics
    val_loss = running_loss / len(val_loader)
    acc = 100 * correct / total

    return val_loss, acc