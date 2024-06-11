# lumbar-classification-rsna-challenge-2024

This repo aims at creating a classifier classifier able to compete for [rsna-2024-lumbar-spine-degenerative-classification kagle challenge](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview)

## Baseline CNN

### Usage

Convert DICOM to nifti : 
`python convert_dataset_dcm2nii.py  --input_folder path/to/train_images`
(make sure `dcm2niix -h` is a legit command before using).
/!\ Processing can last roughly 20min 

In order to launch the training, one can use this command :
`python train.py  --evaluate False --data_csv /paht/to/train.csv --base_dir /path/to/train_nifti --num_epochs 10 --GPU_ID 0`


### Preprocessing

* Data loading looks at nifit images in the folder structure
* The dataset is splited between train patients and test patients (20% test)
* Two object from the class "2D_dataset" are created. They encapsulate the label and the image path.
* At each training epoch, the model sees each 3D image once. Each time the image is randomly :
    - flipped
    - rotated (in a 3° range)
    - Shifted (in a 0.1 range)
    - reframed (if exceeding 1024x1024x64 in size)

### Network Structure

The network used is a r3d_18 from torchvision.video library modified to handle 1 channel images and to output a 2D vector.