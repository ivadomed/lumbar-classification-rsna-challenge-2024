''' utility functions for training the models '''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import math


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
        main_output = model(bags)

        # Calculate losses
        loss = criterion(main_output, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rates
        encoder_scheduler.step()
        other_scheduler.step()

        # Calculate accuracy
        _, predicted = torch.max(main_output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update statistics
        running_loss += loss.item()

        # Update progress bar with learning rates
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%',
            'enc_lr': f'{encoder_scheduler.get_last_lr()[0]:.2e}',
            'oth_lr': f'{other_scheduler.get_last_lr()[0]:.2e}'
        })

    # Calculate epoch statistics
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