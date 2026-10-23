"""
Interactive CLI Demo for Cascaded Ocular Screening
Executes end-to-end inference on any fundus photograph, displaying:
  1. YOLOv8 Landmark & Lesion Localization
  2. Attention U-Net Optic Nerve Head Boundaries
  3. Microvascular Segmentation
  4. Automated Glaucoma & Diabetic Retinopathy Diagnostic Scorecard
"""

import os
import sys
import argparse
import cv2
import json
from pathlib import Path

from pipeline.cascaded_pipeline import CascadedRetinalPipeline
from data.synthetic_generator import SyntheticRetinalGenerator


def main():
    parser = argparse.ArgumentParser(description="Run Cascaded Screening Inference on Retinal Image")
    parser.add_argument("--image", type=str, default=None, help="Path to input fundus image")
    parser.add_argument("--output", type=str, default="outputs/diagnostic_report.png", help="Path to save diagnostic board")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # If no image provided, generate a demo sample
    if args.image is None or not os.path.exists(args.image):
        print("[Demo Setup] No valid input image specified. Generating high-resolution benchmark fundus sample...")
        samples_dir = Path("samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        demo_img_path = samples_dir / "sample_patient_fundus.png"

        gen = SyntheticRetinalGenerator(img_size=(640, 640))
        sample = gen.generate_single_sample(has_lesions=True)
        cv2.imwrite(str(demo_img_path), sample["image"])
        input_image_path = str(demo_img_path)
        print(f"[Demo Setup] Sample patient fundus generated at: {input_image_path}")
    else:
        input_image_path = args.image

    print(f"\n[Processing] Loading patient retinal fundus from: {input_image_path}")
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"[Error] Failed to load image from {input_image_path}")
        sys.exit(1)

    # Initialize cascaded pipeline
    pipeline = CascadedRetinalPipeline(device=args.device)

    print("[Pipeline Stage 1] Running YOLOv8 landmark & lesion detection...")
    print("[Pipeline Stage 2] Running Attention U-Net multi-scale boundary segmentation...")
    print("[Pipeline Stage 3] Computing clinical biomarkers & generating report...")

    results = pipeline.run_inference(img)

    # Save diagnostic board
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pipeline.generate_diagnostic_figure(results, save_path=args.output)
    print(f"[Output] 4-Panel Clinical Diagnostic Board saved to: {args.output}")

    # Save structured clinical JSON
    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(results["clinical_report"], f, indent=4)
    print(f"[Output] Structured JSON Clinical Report saved to: {json_path}")

    # Print summary to console
    report = results["clinical_report"]
    glaucoma = report["glaucoma_screening"]
    dr = report["diabetic_retinopathy_screening"]
    vasc = report["vascular_biometrics"]

    print("\n" + "=" * 65)
    print("           PATIENT RETINAL SCREENING REPORT SUMMARY")
    print("=" * 65)
    print(f" [+] Vertical Cup-to-Disc Ratio (vCDR) : {glaucoma['vertical_cdr']:.3f}")
    print(f" [+] Glaucoma Risk Assessment          : {glaucoma['glaucoma_risk_level']}")
    print(f" [+] ISNT Rule Neuroretinal Rim Status : {glaucoma['isnt_rule_compliance']}")
    print(f" [+] Retinal Vessel Density Coverage   : {vasc['vessel_density']}")
    print(f" [+] Detected Diabetic Microaneurysms  : {dr['detected_hemorrhages']}")
    print(f" [+] Detected Retinal Hard Exudates    : {dr['detected_exudates']}")
    print(f" [+] Diabetic Retinopathy Staging      : {dr['provisional_dr_stage']}")
    print("-" * 65)
    print(f" [!] Clinical Action: {report['overall_clinical_recommendation']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
