# Team Neuropoly: RSNA 2024 Lumbar Spine Degenerative Classification Challenge

## Preprocessing 

:warning: This section is only working with BIDS data. In particular it doesn't work with the RSNA challenge dataset, refere yourself to [README.md](../preprocessing_RSNA_challenge/README.md) for the challenge data. 

The preprocessing is transforming your dataset into a format that we can use for inference. Everything is automated once you start the command. 

### Brefore launching the preprocessing

Before launching it you must follow the instructions in the general [README.md](../README.md) to install totalspineseg and other dependencies.


### Launching the preprocessing

To launch the preprocessing you must run the following command: 

`python preprocessing.py --data PATH_TO_DATA --output PATH_TO_WHERE_YOU_WANT_PREPROCESSED_DATA_TO_BE_STORED`