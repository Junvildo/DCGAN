import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random

seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel, 4, 1, 0),
                        batchNorm2d(channel), nn.ReLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)
    
class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()

        self.block = nn.Sequential(
                convTranspose2d(ch_in, ch_out, 4, 2, 1),
                batchNorm2d(ch_out), nn.ReLU() 
                )
    
    def forward(self, x):
        return self.block(x)
    
class OutBlock(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()

        self.block = nn.Sequential(
                convTranspose2d(ch_in, ch_out, 4, 2, 1),
                nn.Tanh() 
                )
    
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.init = InitLayer(100,1024)
        self.up_block_1 = UpBlock(1024, 512)
        self.up_block_2 = UpBlock(512, 256)
        self.up_block_3 = UpBlock(256, 128)
        self.up_block_4 = OutBlock(128, 3)
        self.block = seq(
            self.init,
            self.up_block_1,
            self.up_block_2,
            self.up_block_3,
            self.up_block_4,
        )

    def forward(self, noise):
        return self.block(noise)
    
class FirstDownBlock(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()

        self.block = seq(
            conv2d(ch_in, ch_out, 3, 2, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()

        self.block = seq(
            conv2d(ch_in, ch_out, 3, 2, 1),
            batchNorm2d(ch_out), nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)
    
class PredBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block = seq(
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.down_block_1 = DownBlock(3,3)
        self.pred_block = PredBlock()
        self.block = seq(
            self.down_block_1,
            self.down_block_1,
            self.down_block_1,
            self.down_block_1,
            self.pred_block
        )

    def forward(self, x):
        return self.block(x)

