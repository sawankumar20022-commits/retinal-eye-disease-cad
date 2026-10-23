"""
Unit Tests for Deep Learning Architectures
Verifies forward pass dimensions, layer connectivity, and parameter shapes.
"""

import unittest
import torch
from models.unet.attention_unet import AttentionUNet, AttentionGate, ResidualConvBlock
from models.unet.unet_standard import StandardUNet


class TestModelArchitectures(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 1
        self.height = 128
        self.width = 128
        self.sample_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)

    def test_standard_unet_forward_pass(self):
        model = StandardUNet(in_channels=1, out_channels=2, features=[16, 32, 64, 128])
        model.eval()
        with torch.no_grad():
            out = model(self.sample_tensor)
        self.assertEqual(out.shape, (self.batch_size, 2, self.height, self.width))

    def test_attention_unet_forward_pass(self):
        model = AttentionUNet(in_channels=1, out_channels=2, features=[16, 32, 64, 128])
        model.eval()
        with torch.no_grad():
            out = model(self.sample_tensor)
        self.assertEqual(out.shape, (self.batch_size, 2, self.height, self.width))

    def test_attention_gate_computation(self):
        # f_g = 64, f_l = 32, f_int = 16
        gate = AttentionGate(f_g=64, f_l=32, f_int=16)
        g = torch.randn(2, 64, 16, 16)
        x = torch.randn(2, 32, 32, 32)
        out = gate(g, x)
        self.assertEqual(out.shape, x.shape)

    def test_residual_conv_block(self):
        block = ResidualConvBlock(in_channels=16, out_channels=32)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))


if __name__ == "__main__":
    unittest.main()
