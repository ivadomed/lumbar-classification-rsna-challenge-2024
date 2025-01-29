# function to randomly apply cutmix or mixup to a batch of images

# importations :
import torch
import numpy as np

# functions : 
def cutmix(batch, alpha):
    new_batch = batch.copy()
    for i in range(len(batch["image"])):

        # Randomly choose another image from the batch
        j = np.random.randint(len(batch["image"]))
        while j == i:
            j = np.random.randint(len(batch["image"]))

        # Cutmix
        lam = 0.4 +0.2*np.random.beta(alpha, alpha) # beta distribution
        mask = torch.zeros_like(batch["image"][i])
        mask[:, :int(batch["image"].shape[2] * lam), :] = 1 # mask

        new_batch["image"][i] = batch["image"][i] * mask + batch["image"][j] * (1 - mask)
        new_batch["label"][i] = lam * batch["label"][i] + (1 - lam) * batch["label"][j]

    return new_batch

def mixup(batch, alpha):
    new_batch = batch.copy()
    for i in range(len(batch["image"])):

        # Randomly choose another image from the batch
        j = np.random.randint(len(batch["image"]))
        while j == i:
            j = np.random.randint(len(batch["image"]))

        # Mixup
        lam = np.random.beta(alpha, alpha)
        new_batch["image"][i] = lam * batch["image"][i] + (1 - lam) * batch["image"][j]
        new_batch["label"][i] = lam * batch["label"][i] + (1 - lam) * batch["label"][j]

    return new_batch

def cutmixup(batch, alpha):
    if np.random.rand() < 0.5:
        return cutmix(batch, alpha)
    else:
        return mixup(batch, alpha)