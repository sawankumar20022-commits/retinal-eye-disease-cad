"""
Biomedical Evaluation Metrics for Segmentation
Computes Dice Similarity Coefficient (DSC), Jaccard (IoU), Sensitivity (Recall),
Specificity, Precision, and Accuracy at clinical threshold levels.
"""

import torch
import numpy as np
from typing import Dict, Union


def dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes binary Dice Similarity Coefficient (DSC).
    """
    pred_bin = (torch.sigmoid(y_pred) > threshold).float()
    pred_flat = pred_bin.view(-1)
    true_flat = y_true.view(-1).float()

    intersection = (pred_flat * true_flat).sum().item()
    cardinality = pred_flat.sum().item() + true_flat.sum().item()
    return (2.0 * intersection + eps) / (cardinality + eps)


def iou_score(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Computes Intersection-over-Union (IoU) / Jaccard Index.
    """
    pred_bin = (torch.sigmoid(y_pred) > threshold).float()
    pred_flat = pred_bin.view(-1)
    true_flat = y_true.view(-1).float()

    intersection = (pred_flat * true_flat).sum().item()
    union = pred_flat.sum().item() + true_flat.sum().item() - intersection
    return (intersection + eps) / (union + eps)


def calculate_segmentation_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> Dict[str, float]:
    """
    Computes comprehensive suite of pixel-level classification and segmentation metrics.
    
    Args:
        y_pred (torch.Tensor): Raw model logits or probabilities.
        y_true (torch.Tensor): Binary ground truth mask.
        threshold (float): Decision boundary threshold (default 0.5).
        eps (float): Numerical stability epsilon.
        
    Returns:
        dict: Containing 'dice', 'iou', 'sensitivity', 'specificity', 'precision', 'accuracy'.
    """
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    # Convert logits to binary prediction
    probs = torch.sigmoid(y_pred) if y_pred.min() < 0.0 or y_pred.max() > 1.0 else y_pred
    p = (probs > threshold).float().view(-1)
    t = y_true.float().view(-1)

    tp = (p * t).sum().item()
    fp = (p * (1.0 - t)).sum().item()
    fn = ((1.0 - p) * t).sum().item()
    tn = ((1.0 - p) * (1.0 - t)).sum().item()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)  # Recall
    specificity = (tn + eps) / (tn + fp + eps)
    precision = (tp + eps) / (tp + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
    }
