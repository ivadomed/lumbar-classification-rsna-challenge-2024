import torch
import torch.nn as nn

# this file aims to define using torch nn all of the blocks to design a 3d convnext

# basic bloc "reverse bottleneck", with a depthwise convolution, a pointwise convolution and a group normalization
# one of the core components of the convnext architecture
class ConvNeXtBlockBase(nn.Module):
    def __init__(self, dim, conv_layer, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = conv_layer(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim) 
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, *range(2, x.ndim), 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, -1, *range(1, x.ndim - 1)) 
        x = shortcut + self.drop_path(x)
        return x
    
class ConvNeXtBlockBaseS(nn.Module):
    def __init__(self, dim, conv_layer, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = conv_layer(dim, dim, kernel_size=5, padding=2, groups=dim) 
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim) 
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, *range(2, x.ndim), 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, -1, *range(1, x.ndim - 1)) 
        x = shortcut + self.drop_path(x)
        return x

# whole network, note that :
# the stem cell, the first layer of the network, with a convolution and a group normalization
# note that here the kernel is bigger and the stride also, compared with ResNet
class ConvNeXtBase(nn.Module):
    def __init__(self, in_chans, num_classes, depths, dims, conv_layer, pool_layer):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            conv_layer(in_chans, dims[0], kernel_size=4, stride=4), # here you cn modify it and put like stride=(1,2,2) for 3D
            nn.GroupNorm(num_groups=1, num_channels=dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=dims[i]),
                conv_layer(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlockBase(dim=dims[i], conv_layer=conv_layer) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.pool_layer = pool_layer

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=self.pool_layer)
        x = self.norm(x)
        x = self.head(x)
        return x

class ConvNeXtAxialBase(nn.Module):
    def __init__(self, in_chans, num_classes, depths, dims, conv_layer, pool_layer):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            conv_layer(in_chans, dims[0], kernel_size=4, stride=(2, 2, 1), padding=1), # here you cn modify it and put like stride=(1,2,2) for 3D
            nn.GroupNorm(num_groups=1, num_channels=dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=dims[i]),
                conv_layer(dims[i], dims[i + 1], kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlockBaseS(dim=dims[i], conv_layer=conv_layer) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.pool_layer = pool_layer

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=self.pool_layer)
        x = self.norm(x)
        x = self.head(x)
        return x

class ConvNeXt3D(ConvNeXtBase):
    def __init__(self, in_chans=1, num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__(in_chans, num_classes, depths, dims, conv_layer=nn.Conv3d, pool_layer=[-1, -2, -3])
    
class ConvNeXtAxial3D(ConvNeXtAxialBase):
    def __init__(self, in_chans=1, num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__(in_chans, num_classes, depths, dims, conv_layer=nn.Conv3d, pool_layer=[-1, -2, -3])

class ConvNeXt2D(ConvNeXtBase):
    def __init__(self, in_chans=1, num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__(in_chans, num_classes, depths, dims, conv_layer=nn.Conv2d, pool_layer=[-1, -2])
