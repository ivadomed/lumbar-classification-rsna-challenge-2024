# Perform inference

## Prerequisite 

Your dataset needs to follow the [BIDS](https://bids.neuroimaging.io/index.html) convention. 
In particular you acquisition names should contain the acquisition orientation (sag for sagittal and ax for axial), and the contrast (T2w or T1w). 
You also need to apply the preprocessing to your data following this [rules](../preprocessing/README.md).

## Perform inference

To perform inference you can either perform inference for separate diseases or all diseases. 

To perform inference for only one disease you can type: 
```bash
python inference_[disease].py --data <PATH_DATA_DIR> --model_dir <PATH_MODEL_DIR>  --output_csv <PATH_TO_CSV>
```

To perform inference for all diseases you can type: 
```bash
python inference.py --data <PATH_DATA_DIR> --model_dir <PATH_MODEL_DIR> --output_csv <PATH_TO_CSV>
```

Warning: if the PATH_TO_CSV refers to an already existing csv it will overwrite the existing csv. 

## Output 

The output of the inference is a csv containing the predictions per subject per disease per level. 
