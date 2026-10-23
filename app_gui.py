"""
Interactive Ophthalmic Diagnostic Dashboard
Provides a graphical desktop interface for clinicians to load fundus images,
trigger cascaded inference, and inspect automated CAD biomarkers.
"""

import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from pathlib import Path

from pipeline.cascaded_pipeline import CascadedRetinalPipeline
from data.synthetic_generator import SyntheticRetinalGenerator


class OphthalmicCADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OcuVision-CAD | Multi-Disease Retinal Screening Dashboard")
        self.root.geometry("1150x750")
        self.root.minsize(950, 650)

        # Initialize Pipeline
        self.status_var = tk.StringVar(value="Initializing Deep Learning Pipeline...")
        self.pipeline = None
        self.current_image = None
        self.current_image_path = None
        self.results = None

        self._build_ui()
        self.root.after(100, self._load_models)

    def _build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#1a237e", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        title = tk.Label(header, text="OcuVision-CAD: Retinal Disease Diagnostic System",
                         fg="white", bg="#1a237e", font=("Helvetica", 16, "bold"))
        title.pack(side=tk.LEFT, padx=20, pady=12)

        subtitle = tk.Label(header, text="Cascaded YOLOv8 + Attention U-Net",
                            fg="#9fa8da", bg="#1a237e", font=("Helvetica", 11))
        subtitle.pack(side=tk.RIGHT, padx=20, pady=15)

        # Control Toolbar
        toolbar = tk.Frame(self.root, bg="#e8eaf6", height=45)
        toolbar.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)

        btn_load = ttk.Button(toolbar, text="📂 Load Fundus Image", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=5, pady=5)

        btn_demo = ttk.Button(toolbar, text="🧪 Generate Benchmark Sample", command=self.generate_sample)
        btn_demo.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_run = ttk.Button(toolbar, text="⚡ Run CAD Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=10, pady=5)

        # Main Content Split
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Panel: Image Canvas
        left_frame = ttk.LabelFrame(main_paned, text=" Diagnostic Image Preview ")
        main_paned.add(left_frame, weight=3)

        self.canvas_label = tk.Label(left_frame, bg="#212121")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right Panel: Clinical Scorecard
        right_frame = ttk.LabelFrame(main_paned, text=" Automated Diagnostic Scorecard ")
        main_paned.add(right_frame, weight=2)

        self.report_text = tk.Text(right_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#fafafa")
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.report_text.insert(tk.END, "Please load a retinal fundus image and click 'Run CAD Analysis'.")
        self.report_text.config(state=tk.DISABLED)

        # Status Bar
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#eeeeee")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_models(self):
        self.status_var.set("Loading Attention U-Net & YOLO architectures...")
        self.root.update_idletasks()
        try:
            self.pipeline = CascadedRetinalPipeline(device="cpu")
            self.status_var.set("Ready. Please load an image to begin.")
        except Exception as e:
            self.status_var.set(f"Error loading models: {e}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.bmp")])
        if path:
            self._display_selected_image(path)

    def generate_sample(self):
        self.status_var.set("Synthesizing clinical test sample...")
        self.root.update_idletasks()
        gen = SyntheticRetinalGenerator(img_size=(512, 512))
        sample = gen.generate_single_sample(has_lesions=True)
        os.makedirs("samples", exist_ok=True)
        demo_path = "samples/gui_sample_fundus.png"
        cv2.imwrite(demo_path, sample["image"])
        self._display_selected_image(demo_path)

    def _display_selected_image(self, path):
        self.current_image_path = path
        self.current_image = cv2.imread(path)
        if self.current_image is not None:
            self._render_image_on_canvas(self.current_image)
            self.btn_run.config(state=tk.NORMAL)
            self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def _render_image_on_canvas(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_img.shape[:2]
        max_h, max_w = 580, 580
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas_label.config(image=self.tk_img)

    def run_analysis(self):
        if self.current_image is None or self.pipeline is None:
            return

        self.status_var.set("Analyzing fundus: Running YOLO detection & Attention U-Net segmentation...")
        self.root.update_idletasks()

        results = self.pipeline.run_inference(self.current_image)
        self.results = results

        # Generate Diagnostic Figure & Display
        diag_path = "outputs/gui_current_report.png"
        os.makedirs("outputs", exist_ok=True)
        self.pipeline.generate_diagnostic_figure(results, save_path=diag_path)

        diag_bgr = cv2.imread(diag_path)
        if diag_bgr is not None:
            self._render_image_on_canvas(diag_bgr)

        # Update Scorecard Text
        report = results["clinical_report"]
        glaucoma = report["glaucoma_screening"]
        dr = report["diabetic_retinopathy_screening"]
        vasc = report["vascular_biometrics"]

        scorecard = (
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

        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, scorecard)
        self.report_text.config(state=tk.DISABLED)

        self.status_var.set("Analysis Complete! Diagnostic scorecard and figure generated.")


def main():
    root = tk.Tk()
    app = OphthalmicCADApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
