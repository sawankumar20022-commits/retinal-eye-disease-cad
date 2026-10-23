"""
Clinically Grounded Synthetic Retinal Fundus Image & Annotation Generator
Generates realistic benchmark fundus datasets with ground truth annotations for:
- YOLOv8 (Optic Disc, Optic Cup, Hemorrhages, Hard Exudates)
- Attention U-Net (Blood Vessels, Dual Optic Disc/Cup masks)

Allows immediate zero-dependency verification, unit testing, and training.
"""

import os
import cv2
import math
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any


class SyntheticRetinalGenerator:
    """
    Simulates anatomical retinal morphology based on ophthalmic geometry:
    - Retinal background with illumination gradient and choroidal textures
    - Optic Nerve Head (Optic Disc + Optic Cup) with ISNT rim dimensions
    - Macula and Foveal Avascular Zone (FAZ)
    - Retinal vascular arcade (superior & inferior temporal arcades)
    - Diabetic retinopathy lesions (hemorrhages, exudates)
    """
    def __init__(self, img_size: Tuple[int, int] = (512, 512), seed: int = 42):
        self.w, self.h = img_size
        random.seed(seed)
        np.random.seed(seed)

    def generate_single_sample(self, has_lesions: bool = True) -> Dict[str, Any]:
        """
        Synthesizes a complete retinal fundus image along with ground truth masks and bounding boxes.
        """
        # 1. Base Circular Retina & Choroidal Background
        fundus = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cx, cy = self.w // 2, self.h // 2
        radius = int(min(self.w, self.h) * 0.46)

        # Draw smooth reddish-orange retinal base
        y_grid, x_grid = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        fov_mask = (dist_from_center <= radius).astype(np.uint8)

        # Retinal color gradient: Orange-red (BGR: [15-30, 45-80, 160-220])
        base_b = np.clip(25 - (dist_from_center / radius * 20), 5, 35).astype(np.uint8)
        base_g = np.clip(70 - (dist_from_center / radius * 35), 20, 95).astype(np.uint8)
        base_r = np.clip(200 - (dist_from_center / radius * 60), 100, 240).astype(np.uint8)

        fundus[:, :, 0] = base_b * fov_mask
        fundus[:, :, 1] = base_g * fov_mask
        fundus[:, :, 2] = base_r * fov_mask

        # Add subtle anatomical noise / choroidal vessel texture
        noise = np.random.normal(0, 4, (self.h, self.w, 3)).astype(np.int16)
        fundus = np.clip(fundus.astype(np.int16) + noise * fov_mask[:, :, None], 0, 255).astype(np.uint8)

        # 2. Optic Disc & Cup Positioning (Nasal side)
        is_right_eye = random.choice([True, False])
        od_offset_x = -int(self.w * 0.22) if is_right_eye else int(self.w * 0.22)
        od_cx = cx + od_offset_x + random.randint(-10, 10)
        od_cy = cy + random.randint(-20, 20)

        od_radius_x = int(self.w * 0.08) + random.randint(-4, 4)
        od_radius_y = int(self.h * 0.09) + random.randint(-4, 4)

        # Optic Cup is inside Disc (CDR usually 0.3 to 0.7)
        cdr = random.uniform(0.35, 0.65)
        oc_radius_x = int(od_radius_x * cdr)
        oc_radius_y = int(od_radius_y * cdr)
        oc_cx = od_cx + random.randint(-3, 3)
        oc_cy = od_cy + random.randint(-3, 3)

        # Create Ground Truth Masks
        mask_od = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_oc = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.ellipse(mask_od, (od_cx, od_cy), (od_radius_x, od_radius_y), 0, 0, 360, 255, -1)
        cv2.ellipse(mask_oc, (oc_cx, oc_cy), (oc_radius_x, oc_radius_y), 0, 0, 360, 255, -1)

        # Render Optic Disc on Fundus (Bright yellowish-pink: BGR ~ [110, 190, 245])
        od_region = mask_od > 0
        fundus[od_region] = [
            int(fundus[od_region, 0].mean() * 0.3 + 120 * 0.7),
            int(fundus[od_region, 1].mean() * 0.3 + 195 * 0.7),
            int(fundus[od_region, 2].mean() * 0.3 + 245 * 0.7),
        ]
        # Render Optic Cup on Fundus (Pale center: BGR ~ [160, 220, 255])
        oc_region = mask_oc > 0
        fundus[oc_region] = [
            int(fundus[oc_region, 0].mean() * 0.2 + 180 * 0.8),
            int(fundus[oc_region, 1].mean() * 0.2 + 230 * 0.8),
            int(fundus[oc_region, 2].mean() * 0.2 + 255 * 0.8),
        ]
        # Smooth boundaries of disc & cup
        fundus = cv2.GaussianBlur(fundus, (3, 3), 0)

        # 3. Fovea & Macula (Opposite side of Optic Disc)
        macula_cx = cx - int(od_offset_x * 0.6)
        macula_cy = cy + random.randint(-5, 5)
        macula_mask = np.zeros((self.h, self.w), dtype=np.float32)
        cv2.circle(macula_mask, (macula_cx, macula_cy), int(self.w * 0.08), 1.0, -1)
        macula_mask = cv2.GaussianBlur(macula_mask, (51, 51), 0)
        # Darken macula region slightly
        for c in range(3):
            fundus[:, :, c] = np.clip(fundus[:, :, c].astype(np.float32) - macula_mask * 28, 0, 255).astype(np.uint8)

        # 4. Retinal Vascular Tree Generation
        mask_vessels = np.zeros((self.h, self.w), dtype=np.uint8)
        self._render_vessel_tree(fundus, mask_vessels, od_cx, od_cy, is_right_eye)

        # 5. Diabetic Retinopathy Lesions (Hemorrhages & Exudates)
        yolo_boxes = []
        # Optic Disc Bounding Box: [class_id, x_min, y_min, x_max, y_max]
        od_x1, od_y1 = max(0, od_cx - od_radius_x), max(0, od_cy - od_radius_y)
        od_x2, od_y2 = min(self.w, od_cx + od_radius_x), min(self.h, od_cy + od_radius_y)
        yolo_boxes.append({"class_id": 0, "name": "optic_disc", "box": [od_x1, od_y1, od_x2, od_y2]})

        # Optic Cup Bounding Box
        oc_x1, oc_y1 = max(0, oc_cx - oc_radius_x), max(0, oc_cy - oc_radius_y)
        oc_x2, oc_y2 = min(self.w, oc_cx + oc_radius_x), min(self.h, oc_cy + oc_radius_y)
        yolo_boxes.append({"class_id": 1, "name": "optic_cup", "box": [oc_x1, oc_y1, oc_x2, oc_y2]})

        if has_lesions and random.random() > 0.3:
            # Add Hemorrhages (class 2)
            num_hems = random.randint(2, 6)
            for _ in range(num_hems):
                hx = cx + random.randint(-int(radius * 0.7), int(radius * 0.7))
                hy = cy + random.randint(-int(radius * 0.7), int(radius * 0.7))
                hr = random.randint(3, 8)
                if fov_mask[hy, hx] and mask_od[hy, hx] == 0:
                    cv2.circle(fundus, (hx, hy), hr, (10, 15, 110), -1) # Dark red
                    yolo_boxes.append({
                        "class_id": 2, "name": "hemorrhage",
                        "box": [max(0, hx - hr - 2), max(0, hy - hr - 2), min(self.w, hx + hr + 2), min(self.h, hy + hr + 2)]
                    })

            # Add Hard Exudates (class 3)
            num_exs = random.randint(2, 5)
            for _ in range(num_exs):
                ex_x = macula_cx + random.randint(-40, 40)
                ex_y = macula_cy + random.randint(-40, 40)
                ex_r = random.randint(2, 5)
                if fov_mask[ex_y, ex_x] and mask_od[ex_y, ex_x] == 0:
                    cv2.circle(fundus, (ex_x, ex_y), ex_r, (150, 230, 255), -1) # Bright yellow
                    yolo_boxes.append({
                        "class_id": 3, "name": "hard_exudate",
                        "box": [max(0, ex_x - ex_r - 2), max(0, ex_y - ex_r - 2), min(self.w, ex_x + ex_r + 2), min(self.h, ex_y + ex_r + 2)]
                    })

        # Apply final FOV circular crop
        fundus = cv2.bitwise_and(fundus, fundus, mask=fov_mask)

        # Dual mask: Channel 0 = Optic Disc, Channel 1 = Optic Cup
        dual_mask = np.stack([mask_od, mask_oc], axis=-1)

        return {
            "image": fundus,
            "mask_vessels": mask_vessels,
            "mask_od": mask_od,
            "mask_oc": mask_oc,
            "dual_mask": dual_mask,
            "fov_mask": fov_mask,
            "yolo_boxes": yolo_boxes,
            "vcdr": round(float((oc_radius_y * 2) / (od_radius_y * 2)), 4),
            "hcdr": round(float((oc_radius_x * 2) / (od_radius_x * 2)), 4)
        }

    def _render_vessel_tree(self, fundus: np.ndarray, mask: np.ndarray, od_cx: int, od_cy: int, is_right_eye: bool):
        """
        Draws anatomical temporal and nasal retinal vascular arcades radiating from optic disc.
        """
        # Direction of temporal arcades towards macula
        direction_x = 1 if is_right_eye else -1

        # 4 Primary Trunks: Superior Temporal, Inferior Temporal, Superior Nasal, Inferior Nasal
        arcades = [
            (direction_x * 0.8, -1.0, 5, 120),  # Superior Temporal
            (direction_x * 0.8, 1.0, 5, 120),   # Inferior Temporal
            (-direction_x * 0.7, -0.6, 4, 90),  # Superior Nasal
            (-direction_x * 0.7, 0.6, 4, 90),   # Inferior Nasal
        ]

        for dx, dy, init_w, max_len in arcades:
            curr_x, curr_y = float(od_cx), float(od_cy)
            curr_w = init_w
            angle = math.atan2(dy, dx)
            
            for step in range(max_len):
                angle += random.uniform(-0.15, 0.15)
                # Curve temporal vessels around macula
                if abs(dx) > 0.7 and step > 20:
                    angle += 0.02 if dy > 0 else -0.02

                curr_x += math.cos(angle) * 2.2
                curr_y += math.sin(angle) * 2.2

                ix, iy = int(curr_x), int(curr_y)
                if not (0 <= ix < self.w and 0 <= iy < self.h):
                    break

                # Draw vessel segment
                vessel_color = (25, 30, 100) # Deep dark red / burgundy
                cv2.circle(fundus, (ix, iy), max(1, int(curr_w)), vessel_color, -1)
                cv2.circle(mask, (ix, iy), max(1, int(curr_w)), 255, -1)

                # Branching offshoots
                if step % 22 == 0 and step > 15:
                    branch_angle = angle + random.choice([-0.6, 0.6])
                    bx, by = curr_x, curr_y
                    bw = max(1, int(curr_w - 1))
                    for _ in range(random.randint(15, 35)):
                        bx += math.cos(branch_angle) * 1.8
                        by += math.sin(branch_angle) * 1.8
                        ibx, iby = int(bx), int(by)
                        if 0 <= ibx < self.w and 0 <= iby < self.h:
                            cv2.circle(fundus, (ibx, iby), int(bw), vessel_color, -1)
                            cv2.circle(mask, (ibx, iby), int(bw), 255, -1)

                curr_w = max(1, curr_w - 0.035)

    def generate_dataset(
        self,
        output_dir: str,
        num_train: int = 40,
        num_val: int = 10,
        num_test: int = 10
    ):
        """
        Creates full directory hierarchy and populates it with synthetic samples,
        U-Net ground truth masks, and YOLO bounding box labels.
        """
        base = Path(output_dir)
        splits = {"train": num_train, "val": num_val, "test": num_test}

        for split, count in splits.items():
            img_dir = base / split / "images"
            mask_dir = base / split / "masks"
            vessel_dir = base / split / "vessels"
            yolo_label_dir = base / "yolo" / "labels" / split
            yolo_img_dir = base / "yolo" / "images" / split

            for d in [img_dir, mask_dir, vessel_dir, yolo_label_dir, yolo_img_dir]:
                d.mkdir(parents=True, exist_ok=True)

            print(f"[Dataset Generator] Generating {count} samples for split '{split}'...")
            for i in range(count):
                sample_id = f"sample_{split}_{i+1:04d}"
                data = self.generate_single_sample(has_lesions=(split != "test" or i % 2 == 0))

                # Save RGB Image
                img_path = img_dir / f"{sample_id}.png"
                cv2.imwrite(str(img_path), data["image"])
                
                # Copy/symlink image to YOLO structure
                cv2.imwrite(str(yolo_img_dir / f"{sample_id}.png"), data["image"])

                # Save U-Net Masks
                cv2.imwrite(str(mask_dir / f"{sample_id}_od_oc.png"), data["dual_mask"])
                cv2.imwrite(str(vessel_dir / f"{sample_id}_vessels.png"), data["mask_vessels"])

                # Save YOLO Labels (.txt normalized: class x_center y_center w h)
                yolo_txt_path = yolo_label_dir / f"{sample_id}.txt"
                with open(yolo_txt_path, "w") as f:
                    for obj in data["yolo_boxes"]:
                        cls_id = obj["class_id"]
                        x1, y1, x2, y2 = obj["box"]
                        # Normalized center and dims
                        xc = ((x1 + x2) / 2.0) / self.w
                        yc = ((y1 + y2) / 2.0) / self.h
                        bw = (x2 - x1) / self.w
                        bh = (y2 - y1) / self.h
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Write YOLO data.yaml configuration file
        yolo_yaml_path = base / "yolo" / "data.yaml"
        yolo_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yolo_yaml_path, "w") as f:
            f.write(f"path: {base.resolve() / 'yolo'}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n\n")
            f.write("names:\n")
            f.write("  0: optic_disc\n")
            f.write("  1: optic_cup\n")
            f.write("  2: hemorrhage\n")
            f.write("  3: hard_exudate\n")

        print(f"[Dataset Generator] Successfully generated full dataset in: {output_dir}")
        print(f"[Dataset Generator] YOLO config created at: {yolo_yaml_path}")
