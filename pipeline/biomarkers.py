"""
Clinical Biomarker Computation & Ophthalmic Diagnostics Engine
Calculates:
1. Vertical & Horizontal Cup-to-Disc Ratio (vCDR, hCDR)
2. ISNT Rule Verification (Inferior >= Superior >= Nasal >= Temporal neuroretinal rim)
3. Retinal Microvascular Density & Tortuosity Index
4. Automated Glaucoma Risk Staging & Diabetic Retinopathy Assessment
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class ClinicalBiomarkerEngine:
    """
    Quantitative diagnostic engine converting segmentation masks and lesion detections
    into standard clinical ophthalmic indicators.
    """
    def __init__(self, glaucoma_cdr_threshold: float = 0.55):
        self.glaucoma_cdr_threshold = glaucoma_cdr_threshold

    def compute_cdr(
        self,
        mask_od: np.ndarray,
        mask_oc: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates Vertical CDR, Horizontal CDR, and Area CDR.
        
        Args:
            mask_od (np.ndarray): Binary mask of Optic Disc (255 or 1).
            mask_oc (np.ndarray): Binary mask of Optic Cup (255 or 1).
            
        Returns:
            dict: Containing vcdr, hcdr, acdr, od_bbox, oc_bbox, and risk classification.
        """
        # Ensure binary uint8
        bin_od = (mask_od > 127).astype(np.uint8)
        bin_oc = (mask_oc > 127).astype(np.uint8)

        # Get contours
        cnt_od, _ = cv2.findContours(bin_od, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_oc, _ = cv2.findContours(bin_oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnt_od or not cnt_oc:
            return {
                "vcdr": 0.0,
                "hcdr": 0.0,
                "acdr": 0.0,
                "glaucoma_risk": "Indeterminate / Optic Disc Not Detected",
                "valid": False
            }

        # Select largest components
        c_od = max(cnt_od, key=cv2.contourArea)
        c_oc = max(cnt_oc, key=cv2.contourArea)

        od_x, od_y, od_w, od_h = cv2.boundingRect(c_od)
        oc_x, oc_y, oc_w, oc_h = cv2.boundingRect(c_oc)

        area_od = cv2.contourArea(c_od)
        area_oc = cv2.contourArea(c_oc)

        # Ratios (clamped to [0.0, 1.0])
        vcdr = min(1.0, oc_h / max(1, od_h))
        hcdr = min(1.0, oc_w / max(1, od_w))
        acdr = min(1.0, area_oc / max(1.0, area_od))

        # Risk Classification
        if vcdr >= 0.65:
            glaucoma_risk = "HIGH RISK (Severe Cupping - Glaucoma Suspect)"
        elif vcdr >= self.glaucoma_cdr_threshold:
            glaucoma_risk = "MODERATE RISK (Borderline Cupping)"
        else:
            glaucoma_risk = "LOW RISK (Physiological Cupping)"

        return {
            "vcdr": round(float(vcdr), 4),
            "hcdr": round(float(hcdr), 4),
            "acdr": round(float(acdr), 4),
            "od_dimensions_px": {"width": od_w, "height": od_h, "area": int(area_od)},
            "oc_dimensions_px": {"width": oc_w, "height": oc_h, "area": int(area_oc)},
            "glaucoma_risk": glaucoma_risk,
            "valid": True
        }

    def verify_isnt_rule(
        self,
        mask_od: np.ndarray,
        mask_oc: np.ndarray,
        is_right_eye: bool = True
    ) -> Dict[str, Any]:
        """
        Verifies compliance with the ISNT Rule for the neuroretinal rim:
        Rim width should satisfy: Inferior >= Superior >= Nasal >= Temporal.
        Deviation indicates focal glaucomatous neuroretinal rim thinning.
        """
        bin_od = (mask_od > 127).astype(np.uint8)
        bin_oc = (mask_oc > 127).astype(np.uint8)

        cnt_od, _ = cv2.findContours(bin_od, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_oc, _ = cv2.findContours(bin_oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnt_od or not cnt_oc:
            return {"isnt_satisfied": False, "status": "Insufficient segmentation"}

        c_od = max(cnt_od, key=cv2.contourArea)
        c_oc = max(cnt_oc, key=cv2.contourArea)

        od_x, od_y, od_w, od_h = cv2.boundingRect(c_od)
        oc_x, oc_y, oc_w, oc_h = cv2.boundingRect(c_oc)

        # Rim thickness calculation:
        # Superior Rim = distance between top of disc and top of cup
        rim_superior = max(0, oc_y - od_y)
        # Inferior Rim = distance between bottom of cup and bottom of disc
        rim_inferior = max(0, (od_y + od_h) - (oc_y + oc_h))
        
        # Nasal and Temporal orientation depends on eye (OD: Right Eye, OS: Left Eye)
        if is_right_eye:
            rim_nasal = max(0, oc_x - od_x)
            rim_temporal = max(0, (od_x + od_w) - (oc_x + oc_w))
        else:
            rim_nasal = max(0, (od_x + od_w) - (oc_x + oc_w))
            rim_temporal = max(0, oc_x - od_x)

        # ISNT rule: Inferior >= Superior >= Nasal >= Temporal
        is_i_ge_s = rim_inferior >= rim_superior * 0.9  # 10% clinical tolerance
        is_s_ge_n = rim_superior >= rim_nasal * 0.9
        is_n_ge_t = rim_nasal >= rim_temporal * 0.9

        isnt_rule_obeyed = is_i_ge_s and is_s_ge_n and is_n_ge_t

        return {
            "isnt_obeyed": isnt_rule_obeyed,
            "rim_thickness_px": {
                "inferior": int(rim_inferior),
                "superior": int(rim_superior),
                "nasal": int(rim_nasal),
                "temporal": int(rim_temporal)
            },
            "sub_checks": {
                "inferior_ge_superior": bool(is_i_ge_s),
                "superior_ge_nasal": bool(is_s_ge_n),
                "nasal_ge_temporal": bool(is_n_ge_t)
            },
            "clinical_interpretation": (
                "ISNT Rule Maintained (Normal Rim Architecture)"
                if isnt_rule_obeyed else
                "ISNT Rule Violated (Suspicious Neuroretinal Rim Notching / Thinning)"
            )
        }

    def compute_vessel_density(
        self,
        vessel_mask: np.ndarray,
        fov_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Quantifies retinal microvascular density (fraction of retina occupied by vessels).
        Vascular dropout occurs in diabetic retinopathy and vascular occlusions.
        """
        bin_vessels = (vessel_mask > 127).astype(np.uint8)
        if fov_mask is not None:
            bin_fov = (fov_mask > 127).astype(np.uint8)
            total_retina_pixels = np.sum(bin_fov)
        else:
            total_retina_pixels = vessel_mask.shape[0] * vessel_mask.shape[1]

        vessel_pixels = np.sum(bin_vessels)
        density = vessel_pixels / max(1, total_retina_pixels)

        return {
            "vessel_pixel_count": int(vessel_pixels),
            "retina_area_pixels": int(total_retina_pixels),
            "vascular_density_pct": round(float(density * 100), 2)
        }

    def generate_full_clinical_report(
        self,
        cdr_info: Dict[str, Any],
        isnt_info: Dict[str, Any],
        vessel_info: Dict[str, Any],
        lesions: list
    ) -> Dict[str, Any]:
        """
        Assembles all biomarkers into a structured diagnostic report.
        """
        num_hems = sum(1 for l in lesions if "hemorrhage" in l.get("class_name", "").lower())
        num_exs = sum(1 for l in lesions if "exudate" in l.get("class_name", "").lower())

        # Diabetic Retinopathy Staging
        if num_hems > 5 or num_exs > 5:
            dr_stage = "Moderate-to-Severe NPDR (Multiple Hemorrhages / Exudates)"
        elif num_hems > 0 or num_exs > 0:
            dr_stage = "Mild NPDR (Early Microvascular Lesions Detected)"
        else:
            dr_stage = "No Apparent Diabetic Retinopathy (No Lesions Identified)"

        return {
            "glaucoma_screening": {
                "vertical_cdr": cdr_info.get("vcdr", 0.0),
                "horizontal_cdr": cdr_info.get("hcdr", 0.0),
                "glaucoma_risk_level": cdr_info.get("glaucoma_risk", "N/A"),
                "isnt_rule_compliance": isnt_info.get("clinical_interpretation", "N/A"),
                "rim_measurements": isnt_info.get("rim_thickness_px", {})
            },
            "vascular_biometrics": {
                "vessel_density": f"{vessel_info.get('vascular_density_pct', 0.0)}%"
            },
            "diabetic_retinopathy_screening": {
                "detected_hemorrhages": num_hems,
                "detected_exudates": num_exs,
                "provisional_dr_stage": dr_stage
            },
            "overall_clinical_recommendation": (
                "URGENT: Comprehensive ophthalmology consult advised for elevated CDR or active lesions."
                if (cdr_info.get("vcdr", 0.0) >= 0.6 or num_hems > 3) else
                "ROUTINE: Annual routine retinal screening recommended."
            )
        }
