import torch
import torch.nn as nn
import torch.nn.functional as F
## implement of V-net
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=2, padding=0, dilation=1)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels, kernel_size=2, padding=0, dilation=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = x + skip  # Element-wise sum with the corresponding skip connection
        x = self.conv(x)
        return x

class VNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(VNet, self).__init__()
        self.in_conv = ConvBlock(n_channels, 16, kernel_size=5, padding=2, dilation=1)
        
        # Downsampling path
        self.down1 = DownConvBlock(16, 32)
        self.down2 = DownConvBlock(32, 64)
        self.down3 = DownConvBlock(64, 128)
        self.down4 = DownConvBlock(128, 256)
        
        # Upsampling path
        self.up1 = UpConvBlock(256, 128)
        self.up2 = UpConvBlock(128, 64)
        self.up3 = UpConvBlock(64, 32)
        self.up4 = UpConvBlock(32, 16)
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder path
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.final_conv(x)

