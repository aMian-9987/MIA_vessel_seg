import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_part import DoubleConv, ConvBlock, Downs, Ups
## writen in 2021 sep, reimplement of qing huang's paper: https://pubmed.ncbi.nlm.nih.gov/30144657/
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.in_conv = ConvBlock(n_channels, 16, kernel_size=7)
        self.down1 = Downs(16, 32)
        self.down2 = Downs(32, 64)
        self.down3 = Downs(64, 128)
        self.down4 = Downs(128, 256)
        self.down5 = Downs(256, 512)

        self.up1 = Ups(512 + 256, 256)
        self.up2 = Ups(256 + 128, 128)
        self.up3 = Ups(128 + 64, 64)
        self.up4 = Ups(64 + 32, 32)
        self.up5 = Ups(32 + 16, 16)

        self.out_conv = nn.Conv3d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        ### skip connections
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        return self.out_conv(x)

# Example: Create a U-Net with 1 input channel and 2 output classes
net = UNet3D(n_channels=1, n_classes=2)