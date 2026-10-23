"""
Integration Tests for Biomarker Engine & Cascaded Pipeline
"""

import unittest
import numpy as np
import cv2
from pipeline.biomarkers import ClinicalBiomarkerEngine
from pipeline.cascaded_pipeline import CascadedRetinalPipeline
from data.synthetic_generator import SyntheticRetinalGenerator


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.biomarker_engine = ClinicalBiomarkerEngine()
        self.generator = SyntheticRetinalGenerator(img_size=(256, 256))

    def test_cdr_calculation_synthetic(self):
        # Create known geometric disc and cup
        mask_od = np.zeros((200, 200), dtype=np.uint8)
        mask_oc = np.zeros((200, 200), dtype=np.uint8)

        # Disc radius = 50 (height = 100), Cup radius = 25 (height = 50) => expected CDR = 0.50
        cv2.circle(mask_od, (100, 100), 50, 255, -1)
        cv2.circle(mask_oc, (100, 100), 25, 255, -1)

        result = self.biomarker_engine.compute_cdr(mask_od, mask_oc)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["vcdr"], 0.50, delta=0.05)
        self.assertAlmostEqual(result["hcdr"], 0.50, delta=0.05)

    def test_isnt_rule_verification(self):
        mask_od = np.zeros((200, 200), dtype=np.uint8)
        mask_oc = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask_od, (100, 100), 50, 255, -1)
        cv2.circle(mask_oc, (100, 100), 25, 255, -1)

        isnt_res = self.biomarker_engine.verify_isnt_rule(mask_od, mask_oc)
        self.assertIn("isnt_obeyed", isnt_res)
        self.assertIn("rim_thickness_px", isnt_res)

    def test_cascaded_inference_end_to_end(self):
        sample = self.generator.generate_single_sample(has_lesions=True)
        pipeline = CascadedRetinalPipeline(device="cpu")
        res = pipeline.run_inference(sample["image"])

        self.assertIn("detections", res)
        self.assertIn("mask_od", res)
        self.assertIn("mask_oc", res)
        self.assertIn("mask_vessels", res)
        self.assertIn("clinical_report", res)
        self.assertIn("vcdr", res["cdr_metrics"])


if __name__ == "__main__":
    unittest.main()
