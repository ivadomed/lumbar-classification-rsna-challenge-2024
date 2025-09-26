# Team Neuropoly: RSNA 2024 Lumbar Spine Degenerative Classification Challenge

## ResNet training 

Our first approach consisted in training three ResNet on the challenge's data to perform our predictions. 
The code has to be used after the preprocessing cf this [README.md](../preprocessing/README.md). 

Once the preprocessing has been applied you can train the models using the three files: 
- `train_nfn.py`
- `train_sas.py`
- `train_scs.py`

To do so the command is the same for the three models: 

`python train_MODEL.py --data PATH_TO_THE_PREPROCESSED_DATA --csv_file PATH_TO/train.csv`

You can modify the hyperparameters in the python files directly. Otherwise we put by default the one that worked best according to our experiments. 