# Team Neuropoly: RSNA 2024 Lumbar Spine Degenerative Classification Challenge

## Baseline experiments

The baseline startegy is <b>One condition, one model.</b>
It's training a ResNet18 model, implemented by Monai.
Hyperparameters and preprocessing parameters can be modified via the `config.yml` file.

## Training and testing

Just run the command `python3 train_and_test.py -config config.yml`
