'File to define the training functions'
# training can be launched using argsparsing see at the en of the file

# Importing the necessary libraries

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from monai.networks.nets import ResNet
import matplotlib.pyplot as plt
from data_preparation import prepare_data
from transforms import get_transforms
import os


# weights of the objective loss function:
weight = torch.tensor([1.0, 2.0, 4.0]).cuda()

# Training function
def train_and_evaluate_model(device, type, data_dir, csv_file, batch_size=4, lr=1e-4, epochs=20, val_split=0.25, layers=[3, 4, 6, 3], wd=1e-4, augment=0):
    # Préparer les données
    data_dir_train = os.path.join(data_dir, 'training')
    data_dir_val = os.path.join(data_dir, 'validation')

    transform=get_transforms()
    data_train = prepare_data(data_dir_train, csv_file, transform, type)
    data_val = prepare_data(data_dir_val, csv_file, transform, type)
    
    # augmentation, for a number of augment times, we add a randomly augmented dataset to the training set
    if augment != 0:
        for i in range(augment):
            data_train_prime = prepare_data(data_dir_train, csv_file, transform, type)
            data_train = ConcatDataset([data_train, data_train_prime])

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    
    # Définir le modèle, la loss function et l'optimiseur
    model = ResNet(
            block="bottleneck",
            layers=layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=2,
            num_classes=3,
            ).cuda()
    
    # hyperparameters
    hyperparameters = {
        'batch_size': batch_size,
        'learning_rate': lr,
        'num_epochs': epochs,
        'val_split': val_split,
        'layers': layers,
        'weight_decay': wd,
        'augment': augment,
        'train_set_size': len(data_train),
        'val_set_size': len(data_val)
    }
    model_name = f"{type}_model_layers_{layers}_epochs_{epochs}_lr_{lr}_augmentation_{augment}_wd_{wd}"

    
    model = model.to(device)
    
    criterion = CrossEntropyLoss(weight=weight)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= wd)

    # Listes pour stocker la perte et l'exactitude
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 

    # Entraînement
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0



        for batch in tqdm(train_loader):
            inputs = batch["combinaison"].cuda()
            labels = batch["label"].cuda()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)


        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]}, Accuracy: {100 * correct_predictions / total_predictions}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                
                
                inputs = batch["combinaison"].cuda()
                labels = batch["label"].cuda()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)


        val_losses.append(val_loss / len(val_loader))
        print(f"Validation Loss: {val_losses[-1]}, Validation Accuracy: {100 * correct_predictions / total_predictions}%")

        if val_losses[-1] < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_losses[-1]:.4f}. Saving model...")
            best_val_loss = val_losses[-1]
            
            # Sauvegarde du modèle
            torch.save(model.state_dict(), f"{model_name}.pth")

    print("Entraînement terminé.")

    # Plotting
    plotting(train_losses, val_losses, hyperparameters, best_val_loss, model_name)


# Plotting function
def plotting(train, val, hyperparameters, best_val_loss, model_name):
    plt.figure(figsize=(15, 7))

    #Training and validation loss graph
    plt.plot(train, label='Train Loss')
    plt.plot(val, label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Add hyperparameters to the graph
    hyperparams_text = '\n'.join([f"{key}: {value}" for key, value in hyperparameters.items()])
    plt.text(0.02, 0.98, f"Hyperparameters:\n{hyperparams_text}", 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Add the best validation loss to the graph
    plt.text(0.98, 0.02, f"Best Validation Loss: {best_val_loss:.4f}", 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', 
            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Save the complete figure with the two graphs
    plt.tight_layout()  # To avoid overlapping graphs
    plt.savefig(f'training_loss_{model_name}.png')
    plt.close()


# Launch training using argsparsing

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run MONAI script for medical image processing.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the data is stored.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file containing dataset information.")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Extract the data directory and CSV file path
    data_dir = args.data_dir
    csv_file = args.csv_file
    

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The data directory '{data_dir}' does not exist.")
        return
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The CSV file '{csv_file}' does not exist.")
        return
    
   # Specify the GPU index (0, 1, 2, ...)
    gpu_id = 0  # Change this to the desired GPU index
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    

    train_and_evaluate_model(device, data_dir, csv_file, batch_size=8, lr=5e-5, epochs=50, val_split=0.25, layers=[3, 4, 6, 3], augment=0, type='canal')
   

if __name__ == "__main__":
    main()