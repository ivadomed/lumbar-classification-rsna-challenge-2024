import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from train_mil import train_model_sas
from inference_mil import run_inference, plot_confusion_matrices, save_predicted_values
from mil_definition import MILmodel, convnext_small
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from prepare_data_mil import prepare_data_sas
import json
import wandb
from tqdm import tqdm


def train_and_evaluate_fold(
    fold_idx,
    train_data_left,
    train_data_right,
    val_data_left,
    val_data_right,
    config,
    device
):
    """
    Train and evaluate a single fold
    """
    # Create dataloaders for this fold
    train_data = ConcatDataset([train_data_left, train_data_right])
    val_data = ConcatDataset([val_data_left, val_data_right])
    
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = MILmodel(
        encoder=convnext_small,
        num_layers=config['num_layers']
    ).to(device)
    
    # Train model
    model = train_model_sas(
        data_dir=config['data_dir'],
        csv_file=config['csv_file'],
        num_epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        encoder_lr=config['encoder_lr'],
        freeze_encoder_epoch=config['freeze_encoder_epoch'],
        encoder_cosine_epochs=config['encoder_cosine_epochs'],
        other_cosine_epochs=config['other_cosine_epochs'],
        eta_min_factor_encoder=config['eta_min_factor_encoder'],
        eta_min_factor_other=config['eta_min_factor_other'],
        aux_loss_weight=config['aux_loss_weight'],
        aux_loss_schedule=config['aux_loss_schedule'],
        num_layers=config['num_layers'],
        device=device
    )
    
    # Run inference
    predictions, labels, probs = run_inference(model, val_loader, device)
    
    return predictions, labels, probs, model


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration dictionary
    config = {
        'data_dir': '/home/ge.polymtl.ca/p121315/duke/public/rsna_challenge/20250212nii_data_splits',
        'csv_file': '/home/ge.polymtl.ca/p121315/duke/public/rsna_challenge/dcom_data/train.csv',
        'epochs': 12,
        'batch_size': 2,
        'learning_rate': 5e-5,
        'encoder_lr': 5e-5,
        'freeze_encoder_epoch': 12,
        'encoder_cosine_epochs': 10,
        'other_cosine_epochs': 6,
        'eta_min_factor_encoder': 0.1,
        'eta_min_factor_other': 0.1,
        'aux_loss_weight': 0,
        'aux_loss_schedule': 'constant',
        'num_layers': 3
    }
    
    # Create results directory
    results_dir = 'folded_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize KFold
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Prepare all data
    train_dir = os.path.join(config['data_dir'], 'training')
    val_dir = os.path.join(config['data_dir'], 'validation')
    
    # Get all data
    train_data_left, train_data_right = prepare_data_sas(train_dir, config['csv_file'], random=True)
    val_data_left, val_data_right = prepare_data_sas(val_dir, config['csv_file'], random=False)
    
    # Combine left and right data for splitting
    all_data_left = list(range(len(train_data_left)))
    all_data_right = list(range(len(train_data_right)))
    
    # Store results for each fold
    fold_results = []
    
    # Train and evaluate each fold
    for fold_idx, (train_idx_left, val_idx_left) in enumerate(kfold.split(all_data_left)):
        print(f"\nTraining Fold {fold_idx + 1}/{n_splits}")
        
        # Get train and val indices for right side (same split)
        train_idx_right, val_idx_right = train_idx_left, val_idx_left
        
        # Create fold-specific datasets
        train_data_left_fold = torch.utils.data.Subset(train_data_left, train_idx_left)
        train_data_right_fold = torch.utils.data.Subset(train_data_right, train_idx_right)
        val_data_left_fold = torch.utils.data.Subset(train_data_left, val_idx_left)
        val_data_right_fold = torch.utils.data.Subset(train_data_right, val_idx_right)
        
        # Train and evaluate
        predictions, labels, probs, model = train_and_evaluate_fold(
            fold_idx,
            train_data_left_fold,
            train_data_right_fold,
            val_data_left_fold,
            val_data_right_fold,
            config,
            device
        )
        
        # Save results for this fold
        fold_dir = os.path.join(results_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Plot confusion matrices
        plot_confusion_matrices(labels, predictions, fold_dir)
        
        # Save predictions
        save_predicted_values(probs, labels, fold_dir)
        
        # Calculate metrics
        accuracy = (predictions == labels).mean() * 100
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probs
        })
        
        # Save model
        torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))
        
        print(f"Fold {fold_idx + 1} Accuracy: {accuracy:.2f}%")
    
    # Calculate and save overall results
    all_predictions = np.concatenate([r['predictions'] for r in fold_results])
    all_labels = np.concatenate([r['labels'] for r in fold_results])
    all_probs = np.concatenate([r['probabilities'] for r in fold_results])
    
    # Plot overall confusion matrix
    plot_confusion_matrices(all_labels, all_predictions, results_dir)
    
    # Save overall predictions
    save_predicted_values(all_probs, all_labels, results_dir)
    
    # Calculate and save overall metrics
    overall_accuracy = (all_predictions == all_labels).mean() * 100
    fold_accuracies = [r['accuracy'] for r in fold_results]
    
    results_summary = {
        'overall_accuracy': overall_accuracy,
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies)
    }
    
    with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\nOverall Results:")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Plot fold accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_splits + 1), fold_accuracies)
    plt.axhline(y=np.mean(fold_accuracies), color='r', linestyle='--', label='Mean')
    plt.fill_between(range(1, n_splits + 1),
                    np.mean(fold_accuracies) - np.std(fold_accuracies),
                    np.mean(fold_accuracies) + np.std(fold_accuracies),
                    color='r', alpha=0.2)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Across Folds')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'fold_accuracies.png'))
    plt.close()


if __name__ == "__main__":
    main()
