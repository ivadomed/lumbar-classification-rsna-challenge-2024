# Team Neuropoly: RSNA 2024 Lumbar Spine Degenerative Classification Challenge

This branch defines a **3 step preprocessing** pipeline used to preprocess data from this Kaggle [Challenge](https://www.kaggle.com/code/abhinavsuri/anatomy-image-visualization-overview-rsna-raids)

# Requirements 

To be able to run all the code you need to have installed totalspineseg from [this](https://github.com/neuropoly/totalspineseg) repo. To do so you can simply do: 
```bash
git clone https://github.com/neuropoly/totalspineseg.git
python3 -m pip install -e totalspineseg
```

We begin with __data in DICOM format__, with various acquisitions (T2w axial and sagittal and T1w sagittal) for different subjects : almost 2000 for the training. 
We aim to first convert this data in a __NIfTI format__, and then rearrange the images in the __BIDS convention__, this is **step one** : **niftification.py**

Then, as our goal is to operate __pathology classfication__ on various parts of every subject's anatomy; __spinal canal stenosis (SCS), subarticular stenosis (SS) and neural foraminal narrowing (NFN)__; we run the totalspineseg model on the T2w sagittal images, to be able to identify and localize the special areas we want to look into. This is **step 2** : **totalspineseg.py**

Finally , we __registrate__ the sagittal segmentation into the axial spaces, to be able to localize in every space, and we __extract the patches__ we're interested in for all of the 5 pathologies we have to classify, at __5 different disks levels (L1/L2, L2/L3, L3/L4, L4/L5, L5/S1)__. This is **step 3** : **extraction.py**

Then we will be able to train and apply the classifying heads from the other branches. 

Files can contain commented lines of code for importations on kaggle notebooks at their beginning. You will need to run those to replicate the preprocesing in a kaggle notebook. 