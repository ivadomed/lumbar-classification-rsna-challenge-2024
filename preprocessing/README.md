# Team Neuropoly: RSNA 2024 Lumbar Spine Degenerative Classification Challenge

## Preprocessing 

:warning: This section is only working with the RSNA data. If you want to apply this on your own dataset you need to modify the preprocessing pipeline. 

The preprocessing is transforming the kaggle dataset into a format that we can use for training. Everything is automated once you start the command. 

### Brefore launching the preprocessing

Before launching it you must follow the instructions in the general [README.md](../README.md) to install totalspineseg and other dependencies.

Otherwise you must also download the data locally by following the instructions [here](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data). 
In short you have to run this command: `kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification `

### Launching the preprocessing

To launch the preprocessing you must run the following command: 

`python preprocessing.py --data PATH_TO_KAGGLE_DATA --output PATH_TO_WHERE_YOU_WANT_PREPROCESSED_DATA_TO_BE_STORED --csv_description PATH_TO_train_series_description.csv`