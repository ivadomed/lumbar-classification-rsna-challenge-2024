"""
This file contains the definitions of the classifier networks used in the project.
"""


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Baseline_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Baseline_CNN, self).__init__()
        self.c3d = models.video.r3d_18(pretrained=True)
        # modify so i takes only 1 channel 3D images 
        self.c3d.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        self.c3d.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.c3d(x)
        x = torch.sigmoid(x)
        return x
    


    

