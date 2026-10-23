"""
Retinal Image Preprocessing Pipeline
Implements specialized medical image enhancement:
1. Green-channel extraction (highest contrast band for hemoglobin absorption).
2. CLAHE (Contrast Limited Adaptive Histogram Equalization).
3. Illumination and luminosity correction (Gaussian background subtraction).
4. Circular Field-of-View (FOV) mask generation.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional


class RetinalPreprocessor:
    """
    Preprocessing and enhancement pipeline for retinal fundus imaging.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5
    ):
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )

    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts the green channel from an RGB/BGR retinal image.
        In fundus photography, green light (540-570nm) is strongly absorbed by hemoglobin,
        providing maximal contrast between retinal vessels, lesions, and background parenchyma.
        """
        if len(image.shape) == 2:
            return image
        # Assuming RGB or BGR - in standard OpenCV read (BGR), index 1 is Green
        return image[:, :, 1]

    def apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """Applies Contrast-Limited Adaptive Histogram Equalization."""
        return self.clahe.apply(gray_image)

    def correct_illumination(self, gray_image: np.ndarray, sigma: int = 35) -> np.ndarray:
        """
        Corrects non-uniform illumination and vignetting artifacts common in fundus cameras
        using Gaussian background estimation and subtraction.
        """
        background = cv2.GaussianBlur(gray_image, (0, 0), sigma)
        # Add constant 128 to keep dynamic range centered
        corrected = cv2.addWeighted(gray_image, 1.0, background, -1.0, 128)
        return corrected

    def generate_fov_mask(self, image: np.ndarray, threshold: int = 15) -> np.ndarray:
        """
        Creates a binary Field-of-View (FOV) mask to isolate the ocular retina
        from outer camera border artifacts.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold dark camera border
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological closing to seal internal vessel shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep only largest connected component (the circular retina)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            fov_mask = np.zeros_like(mask)
            fov_mask[labels == largest_label] = 255
            return fov_mask
        return mask

    def preprocess_for_unet(
        self,
        image: np.ndarray,
        return_tensor: bool = True,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Full pipeline transforming raw fundus image into normalized tensor for U-Net.
        
        Returns:
            tensor (torch.Tensor): Shape (1, 1, H, W) normalized.
            enhanced_np (np.ndarray): Shape (H, W) enhanced uint8 image for visualization.
        """
        # 1. Resize to target size
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # 2. Extract green channel
        green = self.extract_green_channel(resized)

        # 3. Correct non-uniform lighting
        illum_corrected = self.correct_illumination(green)

        # 4. Enhance contrast via CLAHE
        enhanced = self.apply_clahe(illum_corrected)

        # 5. Mask FOV to eliminate outer border contrast artifacts
        fov = self.generate_fov_mask(resized)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=fov)

        if not return_tensor:
            return None, enhanced

        # Convert to FloatTensor [0.0, 1.0] and normalize
        norm_img = enhanced.astype(np.float32) / 255.0
        norm_img = (norm_img - self.normalize_mean) / self.normalize_std

        tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(device)
        return tensor, enhanced
