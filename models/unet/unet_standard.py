"""
Standard U-Net Architecture
Baseline implementation based on Ronneberger et al. (2015).
Used as a benchmark baseline against Attention U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2
    Preserves spatial dimensions via padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Max pooling downsampling followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """
    Upsampling via Transposed Conv or Bilinear Interpolation, followed by skip connection concatenation and DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, dropout_rate: float = 0.0):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential padding discrepancies if input dimensions are odd
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]

        if diff_y > 0 or diff_x > 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension
        merged = torch.cat([skip, x], dim=1)
        return self.conv(merged)


class StandardUNet(nn.Module):
    """
    Standard U-Net for Biomedical Semantic Segmentation.
    
    Args:
        in_channels (int): Input image channels (1 for green channel, 3 for RGB).
        out_channels (int): Target mask classes (e.g. 1 for binary vessel mask, 2 for disc+cup).
        features (list[int]): Feature channels across the hierarchy.
        bilinear (bool): If True, use bilinear upsampling; else use ConvTranspose2d.
        dropout_rate (float): Dropout probability for regularization.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
        bilinear: bool = False,
        dropout_rate: float = 0.1
    ):
        super(StandardUNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder path
        self.inc = DoubleConv(in_channels, features[0], dropout_rate=dropout_rate)
        self.down1 = DownBlock(features[0], features[1], dropout_rate=dropout_rate)
        self.down2 = DownBlock(features[1], features[2], dropout_rate=dropout_rate)
        self.down3 = DownBlock(features[2], features[3], dropout_rate=dropout_rate)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor, dropout_rate=dropout_rate)

        # Decoder path
        self.up1 = UpBlock(features[3] * 2, features[3] // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up2 = UpBlock(features[3], features[2] // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up3 = UpBlock(features[2], features[1] // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up4 = UpBlock(features[1], features[0], bilinear=bilinear, dropout_rate=dropout_rate)

        # Final 1x1 projection convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
