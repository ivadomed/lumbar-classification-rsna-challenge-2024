from torch.utils.data import DataLoader
from Baseline_CNN.data_manager_dcm import Dataset_2D, find_T1w_T2w_paths, dataset_splitter, paths_to_Dataset
from contrast_classifier_Networks import ResNet18SingleChannel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


### Retrieve the arguments
# Create the parser
parser = argparse.ArgumentParser(description='Train and evaluate the model.')

# Add the arguments
parser.add_argument('--model_path', type=str, default='none',
                    help='the path to the model, Random weights by Default')
parser.add_argument('--outputs_directory', type=str, default='outputs',
                    help='the path to the output model, outputs by Default')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
model_path = args.model_path
outputs_directory = args.outputs_directory

# Define the training loop
def testing_loop(model, test_dataset):
    queue_line = np.arange(test_dataset.length)
    np.random.shuffle(queue_line)
    val_predictions = []
    val_labels = []
    index=0
    queue_line = np.repeat(queue_line, 3)
    with torch.no_grad():
        for i in queue_line:
            image, label = test_dataset[i]
            image, label = image.to(device), label.to(device)
            output = model(image)
            prediction = torch.round(output)
            val_predictions.append(prediction.cpu().detach().numpy())
            val_labels.append(label.cpu().detach().numpy())
            # if the prediction is correct save the 2D slice as a png image
            if prediction.cpu().detach().numpy()[0][0] == label.cpu().detach().numpy()[0][0] and prediction.cpu().detach().numpy()[0][1] == label.cpu().detach().numpy()[0][1]:
                # the anme of the file from which the file has been taken 
                file_path = test_dataset.data["paths"][i]
                file_name = file_path.split("\\")[-1].split(".")[0]

                image = image.cpu().detach().numpy()
                image = np.squeeze(image)
                image = np.transpose(image)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(outputs_directory + '//correct_predictions//' + file_name + str(index) + '.png')
            else:
                # the anme of the file from which the file has been taken 
                file_path = test_dataset.data["paths"][i]
                file_name = file_path.split("\\")[-1].split(".")[0]

                image = image.cpu().detach().numpy()
                image = np.squeeze(image)
                image = np.transpose(image)
                image = (image - np.min(image)) / (np.max(image) - np.min(image))
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(outputs_directory + '//incorrect_predictions//' + file_name + str( index) +'.png')
            index += 1

    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)

    accuracy = np.mean(val_predictions == val_labels)

    return accuracy, val_predictions, val_labels
#### Load the data
# Define the base directory
base_dir = "data//data-multi-subject//"

# Find the relative paths of the T1w and T2w files in the specified directory
t1w_file_paths, t2w_file_paths = find_T1w_T2w_paths(base_dir)

# Split the data into training and validation sets
pd_train_data, pd_val_data = dataset_splitter(t1w_file_paths, t2w_file_paths, random_seed=42, train_ratio=0.8)

# Create the training and validation datasets
train_dataset = paths_to_Dataset(pd_train_data)
val_dataset = paths_to_Dataset(pd_val_data, val=True)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = ResNet18SingleChannel(num_classes=2).to(device)
if model_path != 'none':
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded : " + model_path)

#### Evaluate the model
for i in range(1):
    accuracy, val_predictions, val_labels= testing_loop(model, val_dataset)
print(f"Accuracy: {accuracy/(i+1)}")

# plot the confusion matrix

labels_names = ["T1w", "T2w"]
conf_matrix = confusion_matrix([np.argmax(val_labels[i]) for i in range(len(val_labels))], [np.argmax(val_predictions[i]) for i in range(len(val_predictions))])
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=labels_names, yticklabels=labels_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()