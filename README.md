# Team Neuropoly: RSNA 2024 Lumbar Spine Degenerative Classification Challenge

This repos defines the proposed method from Teamn Neuropoly for the [RSNA 2024 Lumbar Spine Degenerative Classification Challenge](https://www.kaggle.com/code/abhinavsuri/anatomy-image-visualization-overview-rsna-raids).

# Citation

> Dagonneau T, Salmona A, Molinier N, Cohen-Adad J. _Automatic radiology assessment of Lumbar Degenerative Diseases_. Proceedings of the 41st Annual Meeting of ESMRMB. Marseille, France 2025

# Requirements 

First create a virtual environment and activate it: 
```bash 
python -m venv .env 
source .env/bin/activate
```

Then install the recquired libraries: 
```bash
pip install -r requirements.txt
```

Then to be able to run all the code you need to have installed totalspineseg from [this](https://github.com/neuropoly/totalspineseg) repo. To do so you can simply do: 
```bash
git clone https://github.com/neuropoly/totalspineseg.git
python3 -m pip install -e totalspineseg
```

# Performing inference

To perform inference you need to download the model. 
Then you have to preprocess your data (they need to be following the [BIDS](https://bids.neuroimaging.io/index.html) convention) by following the instructions in the [preprocessing](preprocessing/README.md) folder. 
Then you can perform inference following the instructions in the [ResNet_inference](ResNet_inference/README.md) folder. 

# Running the code on the challenge data

If you want to replicate our work you first need to preprocess the data from the challenge following the instructions in the [preprocessing](preprocessing/README.md). Then you can train the models using the [ResNet_training](ResNet_training/README.md) folder. 
Then to perform inference on the kaggle dataset you can use the [ResNet_inference_RSNA_challenge](ResNet_inference_RSNA_challenge/README.md).


