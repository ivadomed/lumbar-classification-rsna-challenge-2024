import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from tqdm import tqdm
<<<<<<< HEAD
from prepare_data_mil import prepare_data_scs, prepare_data_sas, prepare_data_nfn, prepare_data_sas_option
from mil_definition import MILmodel, convnext_small
=======
from prepare_data_mil import prepare_data_scs, prepare_data_sas, prepare_data_nfn
from mil_definition import MILmodel
>>>>>>> 6595a9dced291a2b62b4158335f3e1c8169f138a
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import math
import timm 

import timm

convnext_small = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=True, num_classes=0)

# use the challenge's loss function : weighted cross entropy
# with weights 1, 3.5, 7 for the 3 classes, correspond to an imbalance ratio of 1:3.5:7, for sas
weight_challenge = torch.tensor([1.0, 3.5, 7.0]).cuda()

# function that run inference on validation set, and save the loss for each batch, to plot CV
@torch.no_grad()
def run_inference_on_validation_set(model, val_loader, device, criterion):
    losses = []
    model.eval()
    for batch in val_loader:
        bags = batch['bag'].to(device)
        labels = batch['label'].to(device)
        outputs, _ = model(bags)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    var_loss = np.var(losses)
    CV = var_loss / mean_loss
    print(f"CV for severe cases: {CV}, mean loss: {mean_loss}, var loss: {var_loss}")
    return CV, mean_loss, var_loss



class CosineAnnealingStabilizeLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, loop=False, warmup_epochs=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.loop = loop
        self.warmup_epochs = warmup_epochs
        super(CosineAnnealingStabilizeLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.loop:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
        else:
            if self.last_epoch >= self.T_max:
                return [self.eta_min for _ in self.base_lrs]
            
            # Warmup phase
            elif self.last_epoch < self.warmup_epochs:
                return [base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
            
            # Cosine phase (shifted by warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
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
        aux_loss = 1.0
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
        eta_min=encoder_lr * eta_min_factor_encoder,  # Minimum learning rate pour l'encoder
        warmup_epochs=0  # No warmup for this scheduler
    )
    
    other_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * other_cosine_epochs,  # Période spécifique pour le reste
        eta_min=learning_rate * eta_min_factor_other,  # Minimum learning rate pour le reste
        warmup_epochs=0  # No warmup for this scheduler
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
    pretrained_model_path=None,
    warmup_epochs=3,  # New parameter for warmup
    fine_tune_warmup_epochs=2  # New parameter for fine-tuning warmup
):
    """
    Train a single model with a two-phase approach:
    1. Initial training with unweighted loss and single learning rate
    2. Fine-tuning with frozen encoder and weighted loss
    
    Args:
        pretrained_model_path (str, optional): Path to a pretrained model to load and fine-tune.
            If None, a new model will be initialized.
        warmup_epochs (int): Number of epochs for warmup in initial training
        fine_tune_warmup_epochs (int): Number of epochs for warmup in fine-tuning phase
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
            "pretrained_model_path": pretrained_model_path,
            "warmup_epochs": warmup_epochs,
            "fine_tune_warmup_epochs": fine_tune_warmup_epochs
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

    severe_right, severe_left = prepare_data_sas_option(val_dir, csv_file, option=2, random=False)
    severe_loader = DataLoader(ConcatDataset([severe_right, severe_left]), batch_size=2, shuffle=False, num_workers=8)

    moderate_right, moderate_left = prepare_data_sas_option(val_dir, csv_file, option=1, random=False)
    moderate_loader = DataLoader(ConcatDataset([moderate_right, moderate_left]), batch_size=2, shuffle=False, num_workers=8)

    # Loss functions
    unweighted_criterion = nn.CrossEntropyLoss()  # For phase 1 training and validation
    weighted_criterion = nn.CrossEntropyLoss(weight=weight_challenge)  # For phase 2 training and validation


    # Initialize model
    if pretrained_model_path is not None:
        print(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path)
        model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded successfully")
        in_cv, in_mean_loss, in_var_loss = run_inference_on_validation_set(model, severe_loader, device, weighted_criterion)
        print(f"Initial CV for severe cases: {in_cv}, mean loss: {in_mean_loss}, var loss: {in_var_loss}")

    else:
        model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate schedulers
    initial_scheduler = CosineAnnealingStabilizeLR(
        optimizer,
        T_max=len(train_loader) * cosine_epochs,
        eta_min=learning_rate * eta_min_factor,
        warmup_epochs=warmup_epochs
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
    best_severe_loss = float('inf')
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
                    warmup_epochs=fine_tune_warmup_epochs
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
        train_loss, train_acc, _ = train_epoch(
            model, train_loader, criterion, optimizer,
            (scheduler, scheduler), device,
            aux_weight=0,
            epoch=epoch
        )

        # Validate with both criteria
        val_loss, val_acc, _ = validate(
            model, val_loader, unweighted_criterion, device,
            aux_weight=0
        )
        val_weighted_loss, val_weighted_acc, _ = validate(
            model, val_loader, weighted_criterion, device,
            aux_weight=0
        )

        # Store losses for CV calculation
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_weighted_losses.append(val_weighted_loss)

        # Calculate coefficient of variation
        val_severe_cv, val_severe_mean_loss, val_severe_var_loss = run_inference_on_validation_set(model, severe_loader, device, weighted_criterion)
        print(f"CV for severe cases: {val_severe_cv}, mean loss: {val_severe_mean_loss}, var loss: {val_severe_var_loss}")
        val_moderate_cv, val_moderate_mean_loss, val_moderate_var_loss = run_inference_on_validation_set(model, moderate_loader, device, weighted_criterion)


        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_weighted_loss": val_weighted_loss,
            "val_severe_cv": val_severe_cv,
            "val_severe_mean_loss": val_severe_mean_loss,
            "val_moderate_mean_loss": val_moderate_mean_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        # Save best model based on weighted validation loss
        if save_4_wv:
            if val_severe_mean_loss < best_severe_loss and val_loss < 0.5:
                best_severe_loss = val_severe_mean_loss
                best_val_weighted_loss = val_weighted_loss
                best_val_loss = val_loss
                best_val_acc = val_acc
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'initial_scheduler_state_dict': initial_scheduler.state_dict(),
                    'fine_tune_scheduler_state_dict': fine_tune_scheduler.state_dict() if fine_tune_scheduler else None,
                    'val_mean_loss': val_severe_mean_loss,
                    'val_loss': val_loss,
                    'val_weighted_loss': val_weighted_loss,
                    'val_moderate_mean_loss': val_moderate_mean_loss,
                    'val_acc': val_acc,
                    'phase': 2 if epoch >= freeze_encoder_epoch else 1
                }, os.path.join(folder_name, 'best_mil_model.pth'))
                
                # Save config and best loss
                with open(os.path.join(folder_name, 'config.json'), 'w') as f:
                    json.dump(dict(wandb.config), f)
                with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
                    json.dump({
                        'best_val_loss': best_val_loss,
                        'best_val_weighted_loss': best_val_weighted_loss,
                        'best_val_acc': val_acc,
                        'val_severe_cv': val_severe_cv,
                        'val_moderate_mean_loss': val_moderate_mean_loss,
                        'in_cv': in_cv if pretrained_model_path is not None else None
                    }, f)
                
                print(f"Saved new best model with validation loss: {val_loss:.4f} (weighted: {val_weighted_loss:.4f})")

        # else save the model with the validation not weighted loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_weighted_loss = val_weighted_loss
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, os.path.join(folder_name, 'best_mil_model.pth'))
                print(f"Saved new best model with validation loss: {val_loss:.4f}")

                # Save config and best loss
                with open(os.path.join(folder_name, 'config.json'), 'w') as f:
                    json.dump(dict(wandb.config), f)
                with open(os.path.join(folder_name, 'best_loss.json'), 'w') as f:
                    json.dump({
                        'best_val_loss': best_val_loss,
                        'best_val_weighted_loss': best_val_weighted_loss,
                        'best_val_acc': best_val_acc,
                        'val_severe_cv': val_severe_cv, 
                        'in_cv': in_cv if pretrained_model_path is not None else None
                    }, f) 

    wandb.finish()
    return model


def train_model_nfn(
    convnext_small, 
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
    device='cuda',
    
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
    os.makedirs(folder_name, exist_ok=False)

    # Prepare data
    train_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    # Create datasets
    train_data = prepare_data_nfn(train_dir, csv_file, random=True)
    val_data= prepare_data_nfn(val_dir, csv_file, random=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Initialize model
    model = MILmodel(encoder=convnext_small, num_layers=num_layers).to(device)

    # Loss function - CrossEntropyLoss with class weights if needed
    #criterion_encoder = nn.CrossEntropyLoss()
    #criterion_no_encoder = nn.CrossEntropyLoss(weight=weight_challenge)
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
        warmup_epochs=0  # No warmup for this scheduler
    )
    
    other_scheduler = CosineAnnealingStabilizeLR(
        optimizer,  # Pass the entire optimizer
        T_max=len(train_loader) * other_cosine_epochs,  # Période spécifique pour le reste
        eta_min=learning_rate * eta_min_factor_other,  # Minimum learning rate pour le reste
        warmup_epochs=0  # No warmup for this scheduler
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
        #criterion = criterion_encoder
        # Freeze le ConvNext après freeze_encoder_epoch époques
        if epoch >= freeze_encoder_epoch:
            #criterion = criterion_no_encoder
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


def train_model_sas_folds(
    data_dir,
    csv_file,
    num_epochs=20,
    batch_size=8,
    learning_rate=5e-5,
    freeze_encoder_epoch=5,
    cosine_epochs=3,
    eta_min_factor=0.04,
    fine_tune_learning_rate=1e-5,  # Learning rate for fine-tuning phase
    fine_tune_cosine_epochs=5,  # Number of epochs for fine-tuning cosine schedule
    fine_tune_eta_min_factor=0.1,  # Eta min factor for fine-tuning
    num_layers=1,
    device='cuda'
):
    """
    Train 4 models using 4-fold cross-validation with a two-phase approach:
    1. Initial training with unweighted loss and single learning rate
    2. Fine-tuning with frozen encoder and weighted loss
    """
    # Initialize wandb
    wandb.init(
        project="lumbar-mil-sas-folds",
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
            "num_layers": num_layers
        }
    )

    list_encoders = []
    for i in range(4):
        list_encoders.append(timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=True, num_classes=0))

    # Create a folder for all fold models
    folder_name = f"mil_model_sas_folds_{random.randint(0, 1000000)}"
    os.makedirs(folder_name, exist_ok=True)

    # Initialize lists to store results
    fold_results = []
    models = []

    # Load all data once
    print("Loading all data...")
    all_data = []
    for fold in range(4):
        fold_dir = os.path.join(data_dir, str(fold))
        train_data_left, train_data_right = prepare_data_sas(fold_dir, csv_file, random=True)
        all_data.append((train_data_left, train_data_right))

    # Train each fold
    for val_fold in range(4):
        print(f"\nTraining model for fold {val_fold} (using folds {[i for i in range(4) if i != val_fold]} for training)")
        
        # Create training and validation datasets
        train_data_left = ConcatDataset([all_data[i][0] for i in range(4) if i != val_fold])
        train_data_right = ConcatDataset([all_data[i][1] for i in range(4) if i != val_fold])
        val_data_left, val_data_right = all_data[val_fold]

        # Create dataloaders
        train_data = ConcatDataset([train_data_left, train_data_right])
        val_data = ConcatDataset([val_data_left, val_data_right])

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

        # Initialize model
        model = MILmodel(encoder=list_encoders[val_fold], num_layers=num_layers).to(device)

        # Loss functions
        unweighted_criterion = nn.CrossEntropyLoss()  # For phase 1 training and validation
        weighted_criterion = nn.CrossEntropyLoss(weight=weight_challenge)  # For phase 2 training and validation

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate schedulers
        initial_scheduler = CosineAnnealingStabilizeLR(
            optimizer,
            T_max=len(train_loader) * cosine_epochs,
            eta_min=learning_rate * eta_min_factor
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
            print(f"\nFold {val_fold}, Epoch {epoch + 1}/{num_epochs}")

            # Switch to phase 2 if needed
            if epoch >= freeze_encoder_epoch:
                if fine_tune_scheduler is None:
                    # Initialize fine-tuning scheduler
                    fine_tune_scheduler = CosineAnnealingStabilizeLR(
                        optimizer,
                        T_max=len(train_loader) * fine_tune_cosine_epochs,
                        eta_min=fine_tune_learning_rate * fine_tune_eta_min_factor
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
                criterion = unweighted_criterion
                scheduler = initial_scheduler

            # Train
            train_loss, train_acc, _ = train_epoch(
                model, train_loader, criterion, optimizer,
                (scheduler, scheduler), device,
                aux_weight=0,
                epoch=epoch
            )

            # Validate with both criteria
            val_loss, val_acc, _ = validate(
                model, val_loader, unweighted_criterion, device,
                aux_weight=0
            )
            val_weighted_loss, val_weighted_acc, _ = validate(
                model, val_loader, weighted_criterion, device,
                aux_weight=0
            )

            # Store losses for CV calculation
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_weighted_losses.append(val_weighted_loss)

            # Calculate coefficient of variation
            train_cv = np.std(train_losses) / np.mean(train_losses) if len(train_losses) > 1 else 0
            val_cv = np.std(val_losses) / np.mean(val_losses) if len(val_losses) > 1 else 0
            val_weighted_cv = np.std(val_weighted_losses) / np.mean(val_weighted_losses) if len(val_weighted_losses) > 1 else 0

            # Log metrics
            wandb.log({
                f"fold_{val_fold}_epoch": epoch + 1,
                f"fold_{val_fold}_train_loss": train_loss,
                f"fold_{val_fold}_val_loss": val_loss,
                f"fold_{val_fold}_val_weighted_loss": val_weighted_loss,
                f"fold_{val_fold}_train_cv": train_cv,
                f"fold_{val_fold}_val_cv": val_cv,
                f"fold_{val_fold}_val_weighted_cv": val_weighted_cv,
                f"fold_{val_fold}_lr": scheduler.get_last_lr()[0],
                f"fold_{val_fold}_phase": 2 if epoch >= freeze_encoder_epoch else 1
            })

            # Save best model for this fold based on weighted validation loss
            if val_weighted_loss < best_val_weighted_loss:
                best_val_weighted_loss = val_weighted_loss
                best_val_loss = val_loss
                fold_dir = os.path.join(folder_name, f'fold_{val_fold}')
                os.makedirs(fold_dir, exist_ok=True)
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'initial_scheduler_state_dict': initial_scheduler.state_dict(),
                    'fine_tune_scheduler_state_dict': fine_tune_scheduler.state_dict() if fine_tune_scheduler else None,
                    'val_loss': val_loss,
                    'val_weighted_loss': val_weighted_loss,
                    'val_acc': val_acc,
                    'phase': 2 if epoch >= freeze_encoder_epoch else 1
                }, os.path.join(fold_dir, 'best_mil_model.pth'))
                
                # Save config and best loss
                with open(os.path.join(fold_dir, 'config.json'), 'w') as f:
                    json.dump(dict(wandb.config), f)
                with open(os.path.join(fold_dir, 'best_loss.json'), 'w') as f:
                    json.dump({
                        'best_val_loss': best_val_loss,
                        'best_val_weighted_loss': best_val_weighted_loss
                    }, f)
                
                print(f"Saved new best model for fold {val_fold} with validation loss: {val_loss:.4f} (weighted: {val_weighted_loss:.4f})")

        # Store results for this fold
        fold_results.append({
            'fold': val_fold,
            'best_val_loss': best_val_loss,
            'best_val_weighted_loss': best_val_weighted_loss,
            'best_val_acc': val_acc,
            'final_train_cv': train_cv,
            'final_val_cv': val_cv,
            'final_val_weighted_cv': val_weighted_cv
        })
        models.append(model)

    # Calculate and log overall results
    mean_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    mean_val_weighted_loss = np.mean([r['best_val_weighted_loss'] for r in fold_results])
    mean_val_cv = np.mean([r['final_val_cv'] for r in fold_results])
    mean_val_weighted_cv = np.mean([r['final_val_weighted_cv'] for r in fold_results])
    mean_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
    
    wandb.log({
        'mean_val_loss': mean_val_loss,
        'mean_val_weighted_loss': mean_val_weighted_loss,
        'mean_val_cv': mean_val_cv,
        'mean_val_weighted_cv': mean_val_weighted_cv,
        'mean_val_acc': mean_val_acc
    })

    # Save overall results
    with open(os.path.join(folder_name, 'overall_results.json'), 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'mean_val_loss': mean_val_loss,
            'mean_val_weighted_loss': mean_val_weighted_loss,
            'mean_val_cv': mean_val_cv,
            'mean_val_weighted_cv': mean_val_weighted_cv,
            'mean_val_acc': mean_val_acc
        }, f, indent=4)

    wandb.finish()
    return models, fold_results


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    data_dir_simple = '../../duke/public/rsna_challenge/20250212nii_data_splits'
    data_dir = '../../duke/public/rsna_challenge/20250410nii_folds'
    csv_file = '../../duke/public/rsna_challenge/dcom_data/train.csv'

<<<<<<< HEAD
    '''# train using folds
    models, fold_results = train_model_sas_folds(
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=12,
        batch_size=2,
        learning_rate=6e-5,
        freeze_encoder_epoch=6,
        cosine_epochs=10,
        eta_min_factor=0.04,
        fine_tune_learning_rate=2e-5,
        fine_tune_cosine_epochs=5,
        fine_tune_eta_min_factor=0.1,
        num_layers=2,
        device=device
    )'''

    # Train model
    model = train_model_sas(
        data_dir_simple,
        csv_file,
        num_epochs=20,
        batch_size=4,
        learning_rate=6e-5,
        freeze_encoder_epoch=21,
        warmup_epochs=4,
        cosine_epochs=16,
        eta_min_factor=0.3, # below not usefull here
        fine_tune_learning_rate=5e-6,
        fine_tune_cosine_epochs=12,
        fine_tune_eta_min_factor=10.0,
        num_layers=2,
        device=device,
        pretrained_model_path= None, #'/home/ge.polymtl.ca/p121315/rsna_git/lumbar-classification-rsna-challenge-2024/mil_model_sas_992085/best_mil_model.pth',
        save_4_wv=True
=======

    convnext_small = timm.create_model('convnext_small.fb_in22k_ft_in1k_384',
                                   in_chans=1, pretrained=True, num_classes=0)


    # Train model
    model = train_model_nfn(
        convnext_small,
        data_dir=data_dir,
        csv_file=csv_file,
        num_epochs=30,
        batch_size=2,
        learning_rate=5e-3,
        encoder_lr=5e-4,  # Learning rate plus faible pour le ConvNext
        freeze_encoder_epoch=5,  # Freeze le ConvNext après 3 époques
        encoder_cosine_epochs=10,  # Le ConvNext atteint son minimum en 2 époques
        other_cosine_epochs=6,  # Le reste du modèle atteint son minimum en 4 époques
        eta_min_factor_encoder=0.1,  # Le lr de l'encoder descend à 4% de sa valeur initiale
        eta_min_factor_other=0.1,  # Le lr du reste descend à 4% de sa valeur initiale
        aux_loss_weight=0,
        aux_loss_schedule='constant',
        num_layers=1,
        device=device,  
>>>>>>> 6595a9dced291a2b62b4158335f3e1c8169f138a
    )

