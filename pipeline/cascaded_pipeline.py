"""
Cascaded Retinal Deep Learning Pipeline
Integrates:
  Stage 1: YOLOv8 Global Landmark & Lesion Detector
  Stage 2: Attention U-Net High-Resolution ONH (Disc & Cup) & Vessel Segmentation
  Stage 3: Clinical Biomarker Calculation & Diagnostic Visualization
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

from models.yolo.yolo_detector import YOLORetinalDetector
from models.unet.attention_unet import AttentionUNet
from data.preprocessor import RetinalPreprocessor
from pipeline.biomarkers import ClinicalBiomarkerEngine


class CascadedRetinalPipeline:
    """
    End-to-end multi-disease screening pipeline.
    """
    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        unet_od_oc_weights: Optional[str] = None,
        unet_vessel_weights: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.preprocessor = RetinalPreprocessor(target_size=(512, 512))
        self.biomarker_engine = ClinicalBiomarkerEngine()

        # 1. Initialize YOLO Detector
        self.detector = YOLORetinalDetector(weights_path=yolo_weights)

        # 2. Initialize Attention U-Net for Optic Disc & Cup
        self.unet_od_oc = AttentionUNet(in_channels=1, out_channels=2).to(self.device)
        if unet_od_oc_weights and os.path.exists(unet_od_oc_weights):
            self.unet_od_oc.load_state_dict(torch.load(unet_od_oc_weights, map_location=self.device))
        self.unet_od_oc.eval()

        # 3. Initialize Attention U-Net for Vessels
        self.unet_vessels = AttentionUNet(in_channels=1, out_channels=1).to(self.device)
        if unet_vessel_weights and os.path.exists(unet_vessel_weights):
            self.unet_vessels.load_state_dict(torch.load(unet_vessel_weights, map_location=self.device))
        self.unet_vessels.eval()

    def run_inference(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Executes end-to-end cascaded screening on an input fundus image.
        
        Args:
            image (np.ndarray): BGR or RGB fundus image.
            
        Returns:
            dict containing raw predictions, masks, biomarkers, and clinical report.
        """
        h_orig, w_orig = image.shape[:2]

        # STAGE 1: YOLO Detection (Lesions and Optic Nerve Head RoI)
        detections = self.detector.detect(image)
        roi_patch, roi_bbox, od_conf = self.detector.extract_optic_nerve_head_roi(image)
        rx1, ry1, rx2, ry2 = roi_bbox

        # STAGE 2A: Attention U-Net Segmentation on Cropped Optic Nerve Head
        patch_tensor, patch_enhanced = self.preprocessor.preprocess_for_unet(
            roi_patch, return_tensor=True, device=self.device
        )
        with torch.no_grad():
            patch_logits = self.unet_od_oc(patch_tensor)
            patch_probs = torch.sigmoid(patch_logits).squeeze(0).cpu().numpy()

        # Channel 0: Optic Disc, Channel 1: Optic Cup
        raw_mask_od = cv2.resize((patch_probs[0] > 0.5).astype(np.uint8) * 255, (rx2 - rx1, ry2 - ry1))
        raw_mask_oc = cv2.resize((patch_probs[1] > 0.5).astype(np.uint8) * 255, (rx2 - rx1, ry2 - ry1))

        # Project cropped masks back into full image coordinate frame
        full_mask_od = np.zeros((h_orig, w_orig), dtype=np.uint8)
        full_mask_oc = np.zeros((h_orig, w_orig), dtype=np.uint8)
        full_mask_od[ry1:ry2, rx1:rx2] = raw_mask_od
        full_mask_oc[ry1:ry2, rx1:rx2] = raw_mask_oc

        # STAGE 2B: Full-Field Vessel Segmentation
        full_tensor, enhanced_full = self.preprocessor.preprocess_for_unet(
            image, return_tensor=True, device=self.device
        )
        with torch.no_grad():
            vessel_logits = self.unet_vessels(full_tensor)
            vessel_prob = torch.sigmoid(vessel_logits).squeeze().cpu().numpy()

        vessel_mask_resized = cv2.resize((vessel_prob > 0.5).astype(np.uint8) * 255, (w_orig, h_orig))
        fov_mask = self.preprocessor.generate_fov_mask(image)
        vessel_mask = cv2.bitwise_and(vessel_mask_resized, vessel_mask_resized, mask=fov_mask)

        # STAGE 3: Clinical Biomarker Computation
        cdr_info = self.biomarker_engine.compute_cdr(full_mask_od, full_mask_oc)
        isnt_info = self.biomarker_engine.verify_isnt_rule(full_mask_od, full_mask_oc)
        vessel_info = self.biomarker_engine.compute_vessel_density(vessel_mask, fov_mask)
        clinical_report = self.biomarker_engine.generate_full_clinical_report(
            cdr_info, isnt_info, vessel_info, detections
        )

        return {
            "image": image,
            "detections": detections,
            "roi_bbox": roi_bbox,
            "roi_patch": roi_patch,
            "mask_od": full_mask_od,
            "mask_oc": full_mask_oc,
            "mask_vessels": vessel_mask,
            "fov_mask": fov_mask,
            "cdr_metrics": cdr_info,
            "isnt_metrics": isnt_info,
            "vessel_metrics": vessel_info,
            "clinical_report": clinical_report
        }

    def generate_diagnostic_figure(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Creates a high-impact multi-panel clinical diagnostic board.
        """
        image_rgb = cv2.cvtColor(results["image"], cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        plt.subplots_adjust(wspace=0.25, hspace=0.3)

        # Panel 1: YOLO Detections & RoI Bounding Box
        ax1 = axes[0, 0]
        p1_img = image_rgb.copy()
        # Draw RoI box
        rx1, ry1, rx2, ry2 = results["roi_bbox"]
        cv2.rectangle(p1_img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3) # Yellow for RoI
        cv2.putText(p1_img, "Stage-1 ONH RoI", (rx1, max(20, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw lesions
        for d in results["detections"]:
            if "hemorrhage" in d["class_name"].lower():
                bx1, by1, bx2, by2 = d["bbox"]
                cv2.rectangle(p1_img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                cv2.putText(p1_img, "HEM", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif "exudate" in d["class_name"].lower():
                bx1, by1, bx2, by2 = d["bbox"]
                cv2.rectangle(p1_img, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                cv2.putText(p1_img, "EX", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        ax1.imshow(p1_img)
        ax1.set_title("Stage 1: YOLOv8 Landmark & Lesion Localization", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Cropped Optic Nerve Head with Segmented Boundaries
        ax2 = axes[0, 1]
        roi_rgb = cv2.cvtColor(results["roi_patch"], cv2.COLOR_BGR2RGB)
        # Overlay disc and cup contours
        cnt_od, _ = cv2.findContours(
            results["mask_od"][ry1:ry2, rx1:rx2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnt_oc, _ = cv2.findContours(
            results["mask_oc"][ry1:ry2, rx1:rx2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(roi_rgb, cnt_od, -1, (0, 255, 0), 2) # Green for Optic Disc
        cv2.drawContours(roi_rgb, cnt_oc, -1, (255, 0, 0), 2) # Red for Optic Cup

        ax2.imshow(roi_rgb)
        vcdr_val = results["cdr_metrics"].get("vcdr", 0.0)
        ax2.set_title(f"Stage 2: Attention U-Net ONH Boundaries (vCDR: {vcdr_val:.3f})", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # Panel 3: Full-Field Retinal Vessel Segmentation
        ax3 = axes[1, 0]
        vessel_overlay = image_rgb.copy()
        vessel_mask = results["mask_vessels"] > 127
        vessel_overlay[vessel_mask] = [0, 255, 128] # Bright cyan/green vessels
        blended_vessels = cv2.addWeighted(image_rgb, 0.65, vessel_overlay, 0.35, 0)
        ax3.imshow(blended_vessels)
        density_val = results["vessel_metrics"].get("vascular_density_pct", 0.0)
        ax3.set_title(f"Stage 2: Microvascular Segmentation (Density: {density_val}%)", fontsize=11, fontweight="bold")
        ax3.axis("off")

        # Panel 4: Clinical Diagnostic Scorecard
        ax4 = axes[1, 1]
        ax4.axis("off")
        report = results["clinical_report"]
        glaucoma = report["glaucoma_screening"]
        dr = report["diabetic_retinopathy_screening"]
        vasc = report["vascular_biometrics"]

        summary_text = (
            "===========================================\n"
            "        AUTOMATED OPHTHALMIC REPORT        \n"
            "===========================================\n\n"
            f"[GLAUCOMA BIOMARKERS]\n"
            f" - Vertical CDR (vCDR)      : {glaucoma['vertical_cdr']:.3f}\n"
            f" - Horizontal CDR (hCDR)    : {glaucoma['horizontal_cdr']:.3f}\n"
            f" - Risk Stratification      : {glaucoma['glaucoma_risk_level']}\n"
            f" - ISNT Rule Status         : {glaucoma['isnt_rule_compliance']}\n\n"
            f"[DIABETIC RETINOPATHY BIOMARKERS]\n"
            f" - Hemorrhage Count         : {dr['detected_hemorrhages']}\n"
            f" - Hard Exudate Count       : {dr['detected_exudates']}\n"
            f" - Provisional Staging      : {dr['provisional_dr_stage']}\n\n"
            f"[VASCULAR BIOMETRICS]\n"
            f" - Microvascular Density    : {vasc['vessel_density']}\n\n"
            f"[CLINICAL ACTION PLAN]\n"
            f" {report['overall_clinical_recommendation']}\n"
            "==========================================="
        )
        ax4.text(0.05, 0.95, summary_text, fontfamily="monospace", fontsize=10,
                 verticalalignment="top", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f6f9", edgecolor="#b0bec5"))
        ax4.set_title("Stage 3: Automated Clinical Decision Support", fontsize=11, fontweight="bold")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=200)

        return fig
