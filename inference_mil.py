import os
import torch
import json
from torch.utils.data import DataLoader, ConcatDataset
from prepare_data_mil import prepare_data_scs, prepare_data_sas, prepare_data_sas_option
from mil_definition import MILmodel, convnext_small
from train_mil import weight_challenge, run_inference_on_validation_set
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import torch.nn as nn


def plot_confusion_matrices(y_true, y_pred, save_dir):
    """
    Plot two confusion matrices: one with raw counts and one with percentages
    """
    # Créer la matrice de confusion avec les valeurs brutes
    cm = confusion_matrix(y_true, y_pred)
    
    # Créer la figure avec deux subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Premier subplot : valeurs brutes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Deuxième subplot : pourcentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
    plt.close()

def save_predicted_values(predictions, labels, save_dir):
    """
    Save the predicted probabilities and labels to a CSV file
    """
    # normalize the probabilities
    predictions = np.array(predictions)
    predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)

    with open(os.path.join(save_dir, 'predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction', 'label'])
        for pred, label in zip(predictions, labels):
            writer.writerow([pred, label])


@torch.no_grad()
def run_inference(model, val_loader, device):
    """
    Run inference on the validation set
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in tqdm(val_loader, desc='Inference'):
        # Get data
        bags = batch['bag'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        main_output, _ = model(bags)
        
        # Get predictions
        _, predicted = torch.max(main_output, 1)
        
        # Store predictions and labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(main_output.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def run_inference_on_severe_cases(model_path, data_dir, csv_file, device='cuda'):
    """
    Run inference on severe cases using a pretrained model and calculate CV.
    
    Args:
        model_path (str): Path to the pretrained model (.pth file)
        data_dir (str): Directory containing the data
        csv_file (str): Path to the CSV file with labels
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        float: Coefficient of variation for severe cases
    """
    # Load model
    model = MILmodel(encoder=convnext_small, num_layers=2).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Prepare data
    val_dir = os.path.join(data_dir, 'validation')
    val_data_left, val_data_right = prepare_data_sas_severe(val_dir, csv_file, random=False)
    val_data = ConcatDataset([val_data_left, val_data_right])

    severe_loader = DataLoader(val_data, 
                          batch_size=2,
                          shuffle=False,
                          num_workers=8)

    criterion = nn.CrossEntropyLoss(weight=weight_challenge)
    
    # Run inference and calculate CV
    cv, mean_loss, var_loss = run_inference_on_validation_set(model, severe_loader, device, criterion)
    print(f"Coefficient of variation for severe cases: {cv:.4f}, mean loss: {mean_loss:.4f}, var loss: {var_loss:.4f}")
    
    return cv

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    model_dir = '/home/ge.polymtl.ca/p121315/rsna_git/lumbar-classification-rsna-challenge-2024/mil_model_sas247141'
    data_dir = '/home/ge.polymtl.ca/p121315/duke/public/rsna_challenge/20250212nii_data_splits'
    csv_file = '/home/ge.polymtl.ca/p121315/duke/public/rsna_challenge/dcom_data/train.csv'
    
    # Load configuration
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Create validation dataset and dataloader
    val_dir = os.path.join(data_dir, 'validation')
    # val_data = prepare_data_scs(val_dir, csv_file, random=False)
    val_data_l, val_data_r = prepare_data_sas(val_dir, csv_file, random=False)
    val_data = ConcatDataset([val_data_l, val_data_r])
    val_loader = DataLoader(val_data, 
                          batch_size=config['batch_size'],
                          shuffle=False,
                          num_workers=4)
    
    # Initialize model
    model = MILmodel(encoder=convnext_small, num_layers=config['num_layers']).to(device)
    
    # Load model weights
    checkpoint = torch.load(os.path.join(model_dir, 'best_mil_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    # Run inference
    predictions, labels, probs = run_inference(model, val_loader, device)
    
    # Plot and save confusion matrices
    plot_confusion_matrices(labels, predictions, model_dir)
    print(f"Confusion matrices saved to {model_dir}/confusion_matrices.png")

    # Save predicted values
    save_predicted_values(probs, labels, model_dir)
    
    # Calculate and print accuracy
    accuracy = (predictions == labels).mean() * 100
    print(f"\nValidation Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # Example usage
    model_path = "/home/ge.polymtl.ca/p121315/rsna_git/lumbar-classification-rsna-challenge-2024/models/mil_models/mil_model_sas566773/best_mil_model.pth"
    data_dir = '../../duke/public/rsna_challenge/20250212nii_data_splits'
    csv_file = '../../duke/public/rsna_challenge/dcom_data/train.csv'
    
    run_inference_on_severe_cases(model_path, data_dir, csv_file)

