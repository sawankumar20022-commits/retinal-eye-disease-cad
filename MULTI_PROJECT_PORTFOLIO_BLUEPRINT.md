# 🚀 Ocular Computer Vision: 3-Project GitHub Portfolio Roadmap

This blueprint details three complementary, state-of-the-art ophthalmology AI projects spanning the anterior and posterior segments of the human eye. This triad demonstrates mastery across multiple imaging modalities (Fundus Photography, Slit-Lamp Biomicroscopy, and Optical Coherence Tomography - OCT).

---

## 📌 Portfolio Overview

```
                      ┌───────────────────────────────────────────────┐
                      │    OPHTHALMIC COMPUTER VISION PORTFOLIO       │
                      └──────────────────────┬────────────────────────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
┌──────────────────────────────┐┌──────────────────────────────┐┌──────────────────────────────┐
│  PROJECT 1 (Flagship / Done) ││          PROJECT 2           ││          PROJECT 3           │
│       OcuVision-CAD          ││       CorneaTrack-Seg        ││        ChoroidNet-3D         │
│  Dual-Stage Retinal Screening││ Anterior Segment Slit-Lamp   ││ Retinal OCT Volumetric CAD  │
│  YOLOv8 + Attention U-Net    ││ YOLOv8 + Residual U-Net      ││ YOLO B-Scan + U-Net Fluids   │
│  (Glaucoma & Diabetic Ret.)  ││ (Corneal Ulcer & Keratitis)  ││ (AMD & Macular Edema)        │
└──────────────────────────────┘└──────────────────────────────┘└──────────────────────────────┘
```

---

## 🌟 Project 1 (Flagship): OcuVision-CAD
- **Focus:** Posterior Segment (Retina)
- **Modality:** Color Fundus Photography
- **Architectures:** Ultralytics YOLOv8 (Landmarks & Lesions) + Multi-Scale Attention U-Net (Optic Disc/Cup & Blood Vessels)
- **Clinical Targets:**
  - Glaucoma: Automated Vertical Cup-to-Disc Ratio ($\text{vCDR}$) & ISNT Neuroretinal Rim Verification
  - Diabetic Retinopathy: Microaneurysms, Hemorrhages, and Hard Exudates quantification
  - Systemic Microvasculature: Arteriolar density & tortuosity indexing
- **Status:** **Fully implemented with complete source code, tests, and CLI in this repository!**

---

## 🔬 Project 2: CorneaTrack-Seg
- **Focus:** Anterior Segment (Cornea & Anterior Chamber)
- **Modality:** Slit-Lamp Biomicroscopy (Fluorescein Stain & Diffuse White Light)
- **Problem Statement:** Corneal ulcers and bacterial/fungal keratitis are leading causes of preventable blindness in developing countries. Emergency triaging requires rapid localization of epithelial defects and depth profiling.
- **Cascaded AI Pipeline:**
  1. **Stage 1 (YOLOv8 Detection):**
     - Inputs: Diffuse slit-lamp images.
     - Detects the limbus, pupil, and active corneal infiltrates/ulcers with bounding boxes.
     - Classifies ulcer etiology (Bacterial, Fungal, Acanthamoeba, Herpes Simplex).
  2. **Stage 2 (Residual U-Net Segmentation):**
     - Segment the exact stained epithelial defect area (pixel-wise polygon) from cobalt blue fluorescein illumination.
     - Computes the defect percentage relative to corneal area: $\text{Ulcer \%} = \frac{\text{Area}_{\text{Ulcer}}}{\text{Area}_{\text{Cornea}}} \times 100$.
  3. **Stage 3 (Depth Estimation):**
     - Using optical slit-beam cuts, calculates ulcer depth as a percentage of corneal stromal thickness (superficial vs deep stromal melting).
- **Public Datasets to Cite:**
  - SUSTech-SYSU Corneal Ulcer Dataset
  - Harvard Anterior Segment Dataset
- **Key Technical Highlight:** Stain color deconvolution in HSV color space combined with focal Tversky loss to overcome low contrast in marginal keratitis.

---

## 🩺 Project 3: ChoroidNet-3D
- **Focus:** Cross-Sectional Retinal Micromorphology
- **Modality:** Spectral-Domain Optical Coherence Tomography (SD-OCT B-Scans & Volumes)
- **Problem Statement:** Age-Related Macular Degeneration (AMD) and Diabetic Macular Edema (DME) require precise quantification of intraretinal fluid (IRF), subretinal fluid (SRF), and pigment epithelial detachments (PED) across 3D retinal volumes (typically 61–128 B-scans per volume).
- **Cascaded AI Pipeline:**
  1. **Stage 1 (YOLOv8 Slicing & Region Indexing):**
     - Rapidly screens through 61–128 B-scans in $< 0.5$ seconds to flag key slices containing pathology.
     - Bounding boxes around the Foveal depression and Drusen clusters.
  2. **Stage 2 (3D/2.5D Attention U-Net Segmentation):**
     - Multi-class segmentation of 3 distinct pathological fluid compartments:
       - Class 0: Background / Healthy tissue
       - Class 1: Intraretinal Fluid (IRF - cystoid spaces)
       - Class 2: Subretinal Fluid (SRF - neurosensory detachment)
       - Class 3: Pigment Epithelial Detachments (PED / Drusen)
  3. **Stage 3 (Volumetric Biomarker Quantification):**
     - Voxel integration across B-scans to calculate absolute fluid volume in nanoliters ($\text{nL}$).
     - Generates ETDRS (Early Treatment Diabetic Retinopathy Study) 9-sector thickness grid map.
- **Public Datasets to Cite:**
  - RETOUCH Challenge (MICCAI) - OCT Fluid Segmentation Benchmark
  - Kermany OCT Dataset (108,000 OCT B-scans: AMD, DME, Choroidal Neovascularization, Normal)
- **Key Technical Highlight:** Multi-scale dilated convolutions (ASPP) to preserve thin fluid pockets against intense speckle noise in OCT interferometry.

---

## 💼 How to Present This 3-Project Triad in Job Interviews
When an interviewer asks: *"Can you walk me through your Computer Vision projects?"*

**Your Winning Answer:**
> *"During my M.Tech research, I built an end-to-end ophthalmic computer vision portfolio spanning both the anterior and posterior segments of the human eye. My flagship project, **OcuVision-CAD**, tackles the high-resolution dilemma in retinal fundus photography by pioneering a two-stage cascaded architecture: YOLOv8 first pinpoints the optic nerve head and pathological lesions in real-time, feeding an adaptive context-padded patch into a customized Attention U-Net with additive soft attention gates for sub-pixel cup-to-disc ratio (CDR) and vessel segmentation. Building on this, I designed **CorneaTrack-Seg** for anterior segment slit-lamp ulcer staging, and **ChoroidNet-3D** for optical coherence tomography (OCT) fluid volume quantification. Across all projects, I addressed real clinical challenges—such as extreme class imbalance using custom ComboLoss (Focal + Dice), optical wavelength absorption physics, and automated biomarker verification like the clinical ISNT rule."*
