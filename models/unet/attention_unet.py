"""
Attention U-Net Architecture with Residual Convolutions
Reference:
Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018.

Enhancements:
1. Additive Soft Attention Gates (AG) across all skip connections.
2. Residual connections within DoubleConv blocks to prevent gradient dissipation.
3. Optional deep supervision auxiliary heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Additive Soft Attention Gate.
    
    Formula:
        q_att = psi( relu( W_g * g + W_x * x + b_g ) )
        alpha = sigmoid( q_att )
        output = x * alpha
    
    Args:
        f_g (int): Channel dimension of gating signal (coarser scale decoder feature map).
        f_l (int): Channel dimension of skip connection signal (finer scale encoder feature map).
        f_int (int): Intermediate channel bottleneck dimension (usually f_l // 2).
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super(AttentionGate, self).__init__()
        # Linear transformation of gating signal
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        # Linear transformation of skip feature map
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        # Attention coefficient generator
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: Gating signal from decoder (coarser spatial resolution).
        x: Skip connection from encoder (finer spatial resolution).
        """
        # Align spatial dimensions if needed
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        net = self.relu(g1 + x1)
        alpha = self.psi(net)
        
        # Spatial gating: scale feature map x by attention weight alpha
        return x * alpha


class ResidualConvBlock(nn.Module):
    """
    Residual Double Convolution Block:
    x_out = ReLU( Conv(ReLU(Conv(x))) + Shortcut(x) )
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        # Shortcut projection if channel dimension changes
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out


class AttentionUpBlock(nn.Module):
    """
    Decoder block with integrated Attention Gate:
    1. Upsample gating signal.
    2. Pass encoder skip feature through AttentionGate(gating, skip).
    3. Concatenate upsampled signal and gated skip connection.
    4. Pass through ResidualConvBlock.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(AttentionUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ag = AttentionGate(f_g=out_channels, f_l=skip_channels, f_int=skip_channels // 2)
        self.conv = ResidualConvBlock(out_channels + skip_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        
        # Dimension alignment
        diff_y = skip.size()[2] - x_up.size()[2]
        diff_x = skip.size()[3] - x_up.size()[3]
        if diff_y > 0 or diff_x > 0:
            x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2,
                                diff_y // 2, diff_y - diff_y // 2])

        # Filter skip connection via Attention Gate
        gated_skip = self.ag(g=x_up, x=skip)
        
        # Concatenate and pass through convolution block
        merged = torch.cat([gated_skip, x_up], dim=1)
        return self.conv(merged)


class AttentionUNet(nn.Module):
    """
    Multi-Scale Attention U-Net for Retinal Vessel & Optic Disc/Cup Segmentation.
    
    Args:
        in_channels (int): Input image channels (1 for CLAHE Green channel, 3 for RGB).
        out_channels (int): Output segmentation classes (1 for vessels; 2 for Disc & Cup).
        features (list[int]): Encoder feature dimensions [64, 128, 256, 512].
        dropout_rate (float): Dropout probability.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: list = None,
        dropout_rate: float = 0.15
    ):
        super(AttentionUNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder Path
        self.enc1 = ResidualConvBlock(in_channels, features[0], dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ResidualConvBlock(features[0], features[1], dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ResidualConvBlock(features[1], features[2], dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = ResidualConvBlock(features[2], features[3], dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck (Bridge)
        bottleneck_features = features[3] * 2
        self.bottleneck = ResidualConvBlock(features[3], bottleneck_features, dropout_rate=dropout_rate)

        # Decoder Path with Attention Gates
        self.up4 = AttentionUpBlock(bottleneck_features, features[3], features[3], dropout_rate=dropout_rate)
        self.up3 = AttentionUpBlock(features[3], features[2], features[2], dropout_rate=dropout_rate)
        self.up2 = AttentionUpBlock(features[2], features[1], features[1], dropout_rate=dropout_rate)
        self.up1 = AttentionUpBlock(features[1], features[0], features[0], dropout_rate=dropout_rate)

        # Final Pixel Classifier Head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder Feature Extraction
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with Attention-Guided Skip Connections
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Output Logits (pre-sigmoid or pre-softmax)
        logits = self.final_conv(d1)
        return logits
