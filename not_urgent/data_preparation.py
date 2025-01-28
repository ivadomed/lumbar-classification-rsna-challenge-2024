'File to define the functions to prepare the data'

# Importing the necessary libraries

import os
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset

# Conversion dictionnary :
text2int = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}


# Functions :
# globale pipeline returning the dataset
def prepare_data(data_dir, csv_file, transform, type):
    data = []
    labels_df = pd.read_csv(csv_file)
    
    counter = 0
    proportions = [0,0,0]
    
    for subject in os.listdir(data_dir):

        subject_dir = os.path.join(data_dir, subject, 'anat')
        if os.path.isdir(subject_dir):

            for file in os.listdir(subject_dir):

                if type == 'canal':
                    if '_patch.nii.gz' in file and 'foramen' not in file:
                        get_canal(file, subject_dir, subject, csv_file, labels_df, data, counter, proportions)

                if type == 'foramen':
                    if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
                        get_foraminal(file, subject_dir, subject, csv_file, labels_df, data, counter, proportions)
                    
    print(f"Nombre de données chargées: {counter}")
    proportions = [1/(i/counter) for i in proportions]
    print(proportions)
    return Dataset(data=data, transform=transform), proportions

# to get foraminal patches R/L and T1/T2
def get_foraminal(file, subject_dir, subject, csv_file, labels_df, data, counter, proportions):
    if '_patch.nii.gz' in file and 'foramen' in file and 'T1w' in file:
        t1_path = os.path.join(subject_dir, file)
        
        parts = t1_path.split('_')

        disk_level = f"{parts[-5]}_{parts[-4]}"
    
        for t2_file in os.listdir(subject_dir):
            if 'right' in file and 'left' in t2_file: 
                None 
            elif 'left' in file and 'right' in t2_file: 
                None
            else :
                if disk_level in t2_file and 'foramen' in t2_file and 'T2w' in t2_file:
                    t2_path = os.path.join(subject_dir, file)      

        if os.path.exists(t1_path):
            
            # Vérifier la forme de l'image
            t1_image = nib.load(t1_path)
            t2_image = nib.load(t2_path)
            
            t1_image_data = t1_image.get_fdata()
            t2_image_data = t2_image.get_fdata()

            if t1_image_data.ndim == 3 and t2_image_data.ndim == 3 :

                subject_id = (subject.replace('sub-', ''))
                if 'left' in file:
                    label_column = f'left_neural_foraminal_narrowing_{disk_level.lower()}'
                if 'right' in file:
                    label_column = f'right_neural_foraminal_narrowing_{disk_level.lower()}'
                    # Flip the image along the appropriate axis (e.g., flipping along x-axis)
                    t1_image_data = np.flip(t1_image_data, axis=0)  # Flip along the first axis (x-axis)
                    t2_image_data = np.flip(t2_image_data, axis=0)

                # Obtenir l'étiquette brute
                
                label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
                
                # Convertir l'étiquette textuelle en valeur numérique
                label_numeric = text2int.get(label, -1)
                
                if label_numeric != -1:
                    proportions[label_numeric] += 1 
                    counter += 1
                    data.append({"T1": t1_path, "T2": t2_path, "label": label_numeric, "combinaison": None})

# to get central canal patches
def get_canal(file, subject_dir, subject, csv_file, labels_df, data, counter, proportions):
    image_path = os.path.join(subject_dir, file)
                        
    parts = image_path.split('_')
    disk_level = f"{parts[-3]}_{parts[-2]}"

    if os.path.exists(image_path):
        # Vérifier la forme de l'image
        image_data = nib.load(image_path).get_fdata()
        if image_data.ndim == 3:
            subject_id = (subject.replace('sub-', ''))
            
            label_column = f'spinal_canal_stenosis_{disk_level.lower()}'
            # Obtenir l'étiquette brute
            
            label = labels_df.loc[labels_df['study_id'] == subject_id, label_column].values[0]
            
            # Convertir l'étiquette textuelle en valeur numérique
            label_numeric = text2int.get(label, -1)
            if label_numeric != -1:
                counter += 1
                proportions[label_numeric] += 1
                data.append({"image": image_path, "label": label_numeric})

def get_subar(file, subject_dir, subject, csv_file, labels_df, data, counter, proportions):
    image_path = os.path.join(subject_dir, file)
                        
    parts = image_path.split('_')
    disk_level = f"{parts[-3]}_{parts[-2]}"

    if os.path.exists(image_path):
        # Vérifier la forme de l'image
        image_data = nib.load(image_path).get_fdata()
        if image_data.ndim == 3:
            subject_id = (subject.replace('sub-', ''))

            label_column_sasl = f'left_subarticular_stenosis_{disk_level.lower()}'
            label_column_sasr = f'right_subarticular_stenosis_{disk_level.lower()}'
            # Obtenir l'étiquette brute

            label_sasr = labels_df.loc[labels_df['study_id'] == subject_id, label_column_sasl].values[0]
            label_sasl = labels_df.loc[labels_df['study_id'] == subject_id, label_column_sasr].values[0]
            
            # Convertir l'étiquette textuelle en valeur numérique
            label_numeric_sasr = text2int.get(label_sasr, -1)
            label_numeric_sasl = text2int.get(label_sasl, -1)
            if label_numeric_sasr in [0, 1, 2] and label_numeric_sasl in [0, 1, 2]:
                data_right.append({"image": image_path, "label": label_numeric_sasr})
                data_left.append({"image": image_path, "label": label_numeric_sasl})
                counter += 2
            else:
                counter_invalid += 1
                print(f"Étiquette {label_sasr} ou {label_sasl} invalide pour {subject_id} à {disk_level}")