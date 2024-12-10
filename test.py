import matplotlib.pyplot as plt

train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
val_losses = [0.55, 0.45, 0.35, 0.25, 0.30]
best_val_loss = 0.25
epochs = 5
model_name = 'my_model'

hyperparameters = {
    'batch_size': 64,
    'learning_rate': 0.01,
    'num_epochs': 10
}

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Ajouter les hyperparamètres sur le graphique
hyperparams_text = '\n'.join([f"{key}: {value}" for key, value in hyperparameters.items()])
plt.text(0.02, 0.98, f"Hyperparameters:\n{hyperparams_text}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Ajouter la meilleure validation loss sur le graphique
plt.text(0.98, 0.02, f"Best Validation Loss: {best_val_loss:.4f}", 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', 
         horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


# Sauvegarder la figure complète avec les deux graphiques
plt.tight_layout()  # Pour éviter que les graphiques se chevauchent
plt.savefig(f'training_loss_{model_name}.png')
plt.close()