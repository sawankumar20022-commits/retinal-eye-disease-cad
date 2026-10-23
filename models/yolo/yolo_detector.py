"""
YOLOv8/v11 Retinal Landmark & Lesion Detector Wrapper
Integrates Ultralytics YOLO for high-speed anatomical landmark localization
(Optic Disc, Optic Cup) and pathological lesion detection (Hemorrhages, Exudates).
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class YOLORetinalDetector:
    """
    Ocular Landmark and Pathological Lesion Detector powered by YOLO.
    
    Classes:
        0: Optic Disc (OD)
        1: Optic Cup (OC)
        2: Hemorrhage (HEM)
        3: Hard Exudate (EX)
        4: Cotton Wool Spot (CWS)
    """
    CLASS_NAMES = {
        0: "Optic Disc",
        1: "Optic Cup",
        2: "Hemorrhage",
        3: "Hard Exudate",
        4: "Cotton Wool Spot"
    }

    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.25):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads Ultralytics YOLO weights if available; creates fallback detector if not."""
        try:
            from ultralytics import YOLO
            if self.weights_path and os.path.exists(self.weights_path):
                self.model = YOLO(self.weights_path)
            else:
                # Default to nano/small base for fine-tuning or testing
                self.model = YOLO("yolov8n.pt")
            self.has_yolo = True
        except Exception as e:
            print(f"[Warning] YOLO could not be initialized directly: {e}")
            self.has_yolo = False

    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16, project: str = "runs/detect"):
        """
        Launches YOLO fine-tuning on retinal dataset.
        """
        if not self.has_yolo or self.model is None:
            raise RuntimeError("YOLO model is not initialized properly for training.")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name="retinal_yolo_model",
            save=True,
            val=True
        )
        return results

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection inference on a retinal image.
        
        Args:
            image (np.ndarray): Input BGR or RGB image.
            
        Returns:
            list[dict]: Detected bounding boxes with class_id, class_name, confidence, bbox [x1, y1, x2, y2].
        """
        detections = []
        if self.model is not None:
            try:
                results = self.model.predict(source=image, conf=self.conf_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        detections.append({
                            "class_id": cls_id,
                            "class_name": self.CLASS_NAMES.get(cls_id, f"Class_{cls_id}"),
                            "confidence": round(conf, 4),
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        })
                return detections
            except Exception as e:
                print(f"[YOLO Inference Warning] {e}. Falling back to heuristic detector.")

        # Fallback heuristic detector for testing or when weights aren't trained yet
        return self._fallback_detection(image)

    def extract_optic_nerve_head_roi(
        self,
        image: np.ndarray,
        padding_factor: float = 0.40,
        min_crop_size: Tuple[int, int] = (256, 256)
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        Localizes Optic Disc via YOLO and extracts high-resolution padded Region-of-Interest (RoI)
        for stage-2 Attention U-Net segmentation.
        
        Returns:
            cropped_roi (np.ndarray): Padded patch around Optic Nerve Head.
            bbox (tuple): (x1, y1, x2, y2) in original image coordinates.
            confidence (float): Detection confidence.
        """
        h, w = image.shape[:2]
        detections = self.detect(image)
        
        # Filter for Optic Disc (class_id = 0)
        od_candidates = [d for d in detections if d["class_id"] == 0 or "disc" in d["class_name"].lower()]
        
        if od_candidates:
            # Pick highest confidence detection
            best = max(od_candidates, key=lambda x: x["confidence"])
            x1, y1, x2, y2 = best["bbox"]
            conf = best["confidence"]
        else:
            # Fallback: Locate brightest focal region in green channel (Optic Disc is naturally brightest)
            x1, y1, x2, y2, conf = self._heuristic_optic_disc_locator(image)

        # Apply adaptive context padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding_factor)
        pad_y = int(box_h * padding_factor)

        # Enforce minimum size
        pad_x = max(pad_x, (min_crop_size[0] - box_w) // 2) if box_w < min_crop_size[0] else pad_x
        pad_y = max(pad_y, (min_crop_size[1] - box_h) // 2) if box_h < min_crop_size[1] else pad_y

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        cropped_roi = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        return cropped_roi, (crop_x1, crop_y1, crop_x2, crop_y2), conf

    def _heuristic_optic_disc_locator(self, image: np.ndarray) -> Tuple[int, int, int, int, float]:
        """
        Classic computer vision heuristic:
        The Optic Disc corresponds to the maximum intensity cluster in the Green/Red channel.
        """
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            gray = image[:, :, 1] # Green channel has best contrast
        else:
            gray = image

        # Morphological opening and Gaussian blur to eliminate vessel highlights
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        smooth = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        smooth = cv2.GaussianBlur(smooth, (45, 45), 0)

        # Find position of max intensity
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(smooth)
        cx, cy = max_loc

        # Default disc radius ~ 1/10th of image dimension
        radius = max(40, int(min(h, w) * 0.09))
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy + radius)
        return x1, y1, x2, y2, 0.85

    def _fallback_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detections using image processing heuristics."""
        x1, y1, x2, y2, conf = self._heuristic_optic_disc_locator(image)
        # Approximate cup inside disc
        cw = (x2 - x1) * 0.45
        ch = (y2 - y1) * 0.45
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        return [
            {
                "class_id": 0,
                "class_name": "Optic Disc",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            },
            {
                "class_id": 1,
                "class_name": "Optic Cup",
                "confidence": round(conf * 0.9, 4),
                "bbox": [int(cx - cw/2), int(cy - ch/2), int(cx + cw/2), int(cy + ch/2)]
            }
        ]
