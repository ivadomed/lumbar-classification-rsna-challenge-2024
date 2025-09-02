# Perform inference

## Prerequisite 

Your dataset needs to follow the [BIDS](https://bids.neuroimaging.io/index.html) convention. 
In particular you acquisition names should contain the acquisition orientation (sag for sagittal and ax for axial), and the contrast (T2w or T1w). 
You also need to apply the preprocessing to your data following this rules TODODODODO 

## Perform inference

To perform inference you can either perform inference for separate diseases or all diseases. 

To perform inference for only one disease you can type: 
```bash
python inference_[disease].py --data <PATH_DATA_DIR> --model_dir <PATH_MODEL_DIR> 
```

To perform inference for all diseases you can type: 
```bash
python inference_[disease].py --data <PATH_DATA_DIR> --model_dir <PATH_MODEL_DIR> 
```