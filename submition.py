"""
This used a pre_trained network to classify the lumbar spine images from the test dataset.
The results is stored a submission shaped csv file.
"""

from torch.utils.data import DataLoader
from data_manager import Dataset_2D, df_to_Dataset, build_data
from classifier_Networks import C3D
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import argparse
import os
import pandas as pd

# Define the parser
parser = argparse.ArgumentParser(description='Classify lumbar spine images')
parser.add_argument('--data_folder', type=str,  help='Path to the dataset in nifti format')
parser.add_argument('--model_path', type=str, help='Path to the pre-trained model')
parser.add_argument('--GPU_ID', type=int, default=0, help='ID of the GPU to use')

args = parser.parse_args()
data_folder = args.data_folder
model_path = args.model_path
GPU_ID = args.GPU_ID

# Load the data
data_df = build_data(data_folder)
data = df_to_Dataset(data_df, val=True, infer=True)

# Load the model
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
model = C3D(num_classes=75)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Define the data loader
inferded_predictions = []
for i in range(len(data)):
    image, study_id = data[i]
    image = image.to(device)
    output = model(image)
    prediction = torch.round(output)
    inferded_predictions.append({'study_id':study_id, 'prediction':prediction.cpu().detach().numpy()})

#average all precictions with the same study_id
predictions = {}
for pred in inferded_predictions:
    if pred['study_id'] not in predictions:
        predictions[pred['study_id']] = []
    predictions[pred['study_id']].append(pred['prediction'])
for study_id in predictions:
    predictions[study_id] = np.mean(predictions[study_id], axis=0)


# create a dataframe to store the predictions
predictions_df = pd.DataFrame(columns=['row_id','normal_mild','moderate','severe'])

# fill the dataframe with the predictions

issue_list = ['left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1', 'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1']

for study_id in predictions:
    for index, issue_name in enumerate(issue_list):
        predictions_df = predictions_df.append({'row_id':f'{study_id}_{issue_name}', 'normal_mild':predictions[study_id][3*index], 'moderate':predictions[study_id][3*index+1], 'severe':predictions[study_id][3*index+2]}, ignore_index=True)
    
# save the predictions
predictions_df.to_csv('submission.csv', index=False)

        


