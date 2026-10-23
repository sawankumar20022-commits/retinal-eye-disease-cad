"""
Unit Tests for Biomedical Losses & Evaluation Metrics
"""

import unittest
import torch
from models.unet.losses import DiceLoss, BinaryFocalLoss, TverskyLoss, ComboLoss
from models.unet.metrics import dice_coefficient, iou_score, calculate_segmentation_metrics


class TestMetricsAndLosses(unittest.TestCase):
    def test_dice_coefficient_identical_tensors(self):
        # When predicted and true match perfectly (logits high positive)
        logits = torch.ones(1, 1, 64, 64) * 10.0
        target = torch.ones(1, 1, 64, 64)
        score = dice_coefficient(logits, target)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_dice_coefficient_disjoint_tensors(self):
        # When predicted is negative and target is 1
        logits = torch.ones(1, 1, 64, 64) * -10.0
        target = torch.ones(1, 1, 64, 64)
        score = dice_coefficient(logits, target)
        self.assertLess(score, 0.05)

    def test_iou_score_identical_tensors(self):
        logits = torch.ones(1, 1, 64, 64) * 10.0
        target = torch.ones(1, 1, 64, 64)
        score = iou_score(logits, target)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_combo_loss_gradient_flow(self):
        criterion = ComboLoss(focal_weight=0.5, dice_weight=0.5)
        logits = torch.randn(2, 2, 32, 32, requires_grad=True)
        targets = torch.randint(0, 2, (2, 2, 32, 32)).float()

        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_tversky_loss(self):
        criterion = TverskyLoss(alpha=0.7, beta=0.3)
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = criterion(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
