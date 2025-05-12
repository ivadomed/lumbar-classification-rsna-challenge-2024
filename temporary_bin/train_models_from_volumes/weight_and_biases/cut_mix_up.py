# function to randomly apply cutmix or mixup to a batch of images

# importations :
import torch
import numpy as np

# functions : 
def cutmix(batch, alpha):
    new_images = torch.zeros_like(batch["image"])
    new_labels = torch.zeros_like(batch["label"])

    for i in range(len(batch["image"])):

        # Randomly choose another image from the batch
        j = np.random.randint(len(batch["image"]))
        while j == i:
            j = np.random.randint(len(batch["image"]))

        # Cutmix
        lam = 0.4 +0.2*np.random.beta(alpha, alpha) # beta distribution
        mask = torch.zeros_like(batch["image"][i])
        mask[:, :int(batch["image"].shape[2] * lam), :] = 1 # mask

        new_images[i] = batch["image"][i] * mask + batch["image"][j] * (1 - mask)
        new_labels[i] = lam * batch["label"][i] + (1 - lam) * batch["label"][j]
    
    new_batch = {"image": new_images, "label": new_labels}

    return new_batch

def cutmix_indexed(batch, alpha, indexes):
    new_images = torch.zeros_like(batch["image"])
    new_labels = torch.zeros_like(batch["label"])

    for i in range(len(batch["image"])):

        # Randomly choose another image from the batch
        if i in indexes:
            j = np.random.randint(len(batch["image"]))
            while j == i:
                j = np.random.randint(len(batch["image"]))
        else:
            k = np.random.randint(len(indexes))
            j = indexes[k]

        # Cutmix
        lam = 0.4 +0.2*np.random.beta(alpha, alpha) # beta distribution
        mask = torch.zeros_like(batch["image"][i])
        mask[:, :int(batch["image"].shape[2] * lam), :] = 1 # mask

        new_images[i] = batch["image"][i] * mask + batch["image"][j] * (1 - mask)
        new_labels[i] = lam * batch["label"][i] + (1 - lam) * batch["label"][j]
    
    new_batch = {"image": new_images, "label": new_labels}

    return new_batch

def mixup(batch, alpha):
    new_images = torch.zeros_like(batch["image"])
    new_labels = torch.zeros_like(batch["label"])

    for i in range(len(batch["image"])):

        # Randomly choose another image from the batch
        j = np.random.randint(len(batch["image"]))
        while j == i:
            j = np.random.randint(len(batch["image"]))

        # Mixup
        lam = np.random.beta(alpha, alpha)
        new_images[i] = lam * batch["image"][i] + (1 - lam) * batch["image"][j]
        new_labels[i] = lam * batch["label"][i] + (1 - lam) * batch["label"][j]

    new_batch = {"image": new_images, "label": new_labels}

    return new_batch

def cutmixup(batch, alpha):
    if np.random.rand() < 0.5:
        return cutmix(batch, alpha)
    else:
        return mixup(batch, alpha)
    
def cut_if_unhealthy(batch, alpha, prob = 0.5):

    if batch["label"][:, 1:].sum() > 1:
        if np.random.rand() < prob:
            indexes = (batch["label"][:, 1:].sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            return cutmix_indexed(batch, alpha, indexes)
        else:
            return batch
        
    else:
        return batch