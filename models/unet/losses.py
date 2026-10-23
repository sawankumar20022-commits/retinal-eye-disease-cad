"""
Advanced Biomedical Loss Functions for Retinal Segmentation
Addresses severe class imbalance inherent in micro-vessel and optic cup boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for Binary or Multi-Label Semantic Segmentation.
    Computes spatial overlap: 1 - (2 * |X ∩ Y| + eps) / (|X| + |Y| + eps)
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) raw unnormalized logits
        targets: (B, C, H, W) ground truth binary masks (0 or 1)
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_score.mean()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights easy background negatives and emphasizes hard vessel boundaries.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        # p_t is prob of true class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss (Salehi et al., 2017)
    Generalization of Dice coefficient allowing asymmetric penalization
    of False Negatives vs False Positives (crucial for thin capillaries).
    Setting alpha > beta penalizes false negatives heavily.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), probs.size(1), -1)
        t = targets.view(targets.size(0), targets.size(1), -1)

        tp = (p * t).sum(dim=2)
        fp = (p * (1.0 - t)).sum(dim=2)
        fn = ((1.0 - p) * t).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1.0 - tversky).mean()


class ComboLoss(nn.Module):
    """
    Hybrid ComboLoss: Weighted sum of Focal Loss and Dice Loss.
    Combines smooth gradient optimization (Focal) with direct overlap maximization (Dice).
    
    Formula:
        Loss = w_focal * FocalLoss + w_dice * DiceLoss
    """
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        smooth: float = 1e-6
    ):
        super(ComboLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.focal_weight * loss_focal + self.dice_weight * loss_dice
