# fMRI Processing Pipeline

A lightweight, end-to-end pipeline for fMRI preprocessing using **fMRIPrep** with parallel processing, connectivity matrix generation, CPU/GPU flexibility, and comprehensive quality control.

**Features:**  
BIDS compatibility → fMRIPrep preprocessing → Connectivity matrices + Visualization → Parallel processing → Resource monitoring → QC evaluation → Queue management.

---

## Prerequisites

- **Python ≥ 3.8**  
- **Singularity/Apptainer** for containerized execution  
- **Python packages:** `numpy`, `pandas`, `nilearn`, `nibabel`, `tqdm`, `matplotlib`, `seaborn`  
- **fMRIPrep Singularity image:** `fmriprep_latest.sif` (auto-downloaded)  
- **FreeSurfer license:** Valid license file  
- **(Optional)** GPU support with CUDA drivers  
- **(Optional)** conda for environment management  

### Example Environment Setup
```bash
conda create -y -n fmri python=3.10
conda activate fmri
pip install numpy pandas nilearn nibabel tqdm matplotlib seaborn
```

---

## Configuration

All paths and behavior are controlled by `config.py` (shared by all scripts).  
The pipeline is **config-driven** with minimal CLI flags for convenience.

**Paths:**
- `INPUT_DIR`: BIDS input directory  
- `OUTPUT_DIR`: fMRIPrep outputs  
- `WORK_DIR`: Temporary working directory  
- `LOG_DIR`: Pipeline logs  

**Hardware:**
- `USE_GPU`: Enable/disable GPU acceleration  
- `GPUS`: List of available GPU IDs  
- `MAX_PARALLEL`: Maximum parallel subjects  
- `MEMORY_MB`, `NPROCS`, `OMP_NTHREADS`: Resource allocation  

**fMRIPrep:**
- `OUTPUT_SPACES`: Target output spaces  
- `FS_NO_RECONALL`: Skip FreeSurfer reconstruction  
- `LONGITUDINAL`: Enable longitudinal processing  

**Connectivity:**
- `GENERATE_CONNECTIVITY`: Enable connectivity matrix generation  
- `DEFAULT_ATLASES`: Brain atlases for parcellation  
- `DEFAULT_CONFOUNDS`: Confound regressors  
- `CONNECTIVITY_OUTPUT_DIR`: Output directory for matrices  

**QC thresholds:**
- `MIN_VOLUMES_PASS`: 180 (minimum valid volumes for PASS)  
- `MIN_VOLUMES_FAIL`: 120 (below this = FAIL)  
- `FD_PASS_THRESHOLD`: 0.3 mm (mean FD)  
- `FD_WARN_THRESHOLD`: 0.5 mm (mean FD)  
- `HIGH_MOTION_PASS_RATIO`: 20% (frames with FD > 0.5 mm)  
- `HIGH_MOTION_WARN_RATIO`: 30% (frames with FD > 0.5 mm)  

---

## Processing Pipeline

### 1) Setup & Prerequisites Check, `launcher.sh --check`
```bash
./launcher.sh --check
```

### 2) Subject Discovery, `subject_manager.py`
```bash
python subject_manager.py --action discover --output subjects.txt
python subject_manager.py --action add --subjects sub-01 sub-02
python subject_manager.py --action remove --subjects sub-01
```

### 3) fMRI Preprocessing, `pipeline_runner.py`
**Includes automatic connectivity generation and visualization.**

Runs fMRIPrep with:
- BIDS validation and anatomical + functional preprocessing
- Confounds extraction and output generation
- **Automatic connectivity matrix generation** (if `GENERATE_CONNECTIVITY = True`)
- **Automatic visualization generation** (heatmaps + statistics)

```bash
# Single subject (includes fMRIPrep + Connectivity + Visualization)
python pipeline_runner.py --mode single --subject ON00400 --cpu-only

# Parallel processing (includes automatic connectivity + visualization)
python pipeline_runner.py --mode parallel --subjects ON00400 ON01016

# Watch mode (queue processing)
python pipeline_runner.py --mode watch
```

**Note:** All modes automatically include connectivity generation and visualization unless disabled with `--no-connectivity` flag.

### 4) Connectivity Matrix Generation + Visualization, `connectivity_matrix.py`
Automatically runs after successful fMRIPrep completion (if `GENERATE_CONNECTIVITY = True`).

- Extracts timeseries using multiple atlases (AAL, Harvard-Oxford, Schaefer, etc.)
- Computes connectivity matrices (correlation, partial correlation, tangent, covariance)
- **Generates automatic visualizations** (connectivity heatmaps + distribution plots + statistics)
- Saves matrices, timeseries, and visualizations

**Outputs:**
- `sub-*_*_timeseries.csv`: Regional timeseries
- `sub-*_*_*_connectivity.csv`: Connectivity matrices  
- **`sub-*_*_*_visualization.png`: Connectivity heatmaps with statistics**

```bash
# Standalone connectivity generation + automatic visualization
python connectivity_matrix.py --subjects sub-01 sub-02 --atlases aal schaefer-100
python run_connectivity.py --all-completed --atlases aal harvard_oxford
```

**Note:** When using `pipeline_runner.py`, connectivity generation (including visualization) happens automatically after each subject completes fMRIPrep successfully.

### 4.1) Special Parcellation Helper (parcellation_helper_special.py)
A lightweight helper script for dataset-specific or experimental post-processing cases (e.g., multi-session datasets or special connectivity analyses).
This script extends the main pipeline by supporting:

- Automatic selection of the best fMRI run per subject (based on number of timepoints)
- Multi-session handling
- Optional fast I/O by copying large files to local temporary storage
- Connectivity matrix generation across multiple atlases (e.g., DiFuMo, AAL, Harvard-Oxford)

This script is intended for special use cases that require manual control beyond the main pipeline_runner.py.
It is fully compatible with outputs from fMRIPrep and saves timeseries and connectivity matrices in the same standardized format as the main pipeline.
```bash
python scripts/parcellation_helper_special.py \
    --atlases difumo-1024 \
    --subjects_file subjects.txt \
    --root_dir /path/to/bids/derivatives \
    --save_dir /path/to/connectivity \
    --use-temp-storage \
    --temp-dir /tmp/fmri_fastio
```



### 5) Quality Control, `fmri_qc.py`
- Extracts **framewise displacement (FD)** and motion metrics  
- Computes: Mean FD, Max FD, High-Motion Ratio, Volumes  
- Labels runs as **PASS / REVIEW / FAIL**  
- Links to connectivity matrices and visualizations

```bash
# Run QC for all subjects
python fmri_qc.py --all

# Export summary
python fmri_qc.py --all --export-summary
```

---

## Pipeline Integration Features

✅ **Unified Workflow:** Single command runs fMRIPrep → Connectivity → Visualization automatically  
✅ **Smart Processing:** Only generates connectivity for successfully completed fMRIPrep subjects  
✅ **Automatic Visualization:** Every connectivity matrix includes heatmap + statistics plots  
✅ **Resource Optimization:** Connectivity generation uses the same resource allocation as fMRIPrep  
✅ **Error Resilience:** Pipeline continues if connectivity generation fails for individual subjects  
✅ **Duplicate Prevention:** Skips connectivity generation if matrices already exist  
✅ **Configuration Control:** Enable/disable via `GENERATE_CONNECTIVITY` in config.py  

---

## Quick Start

```bash
# 1) Setup environment
conda create -y -n fmri python=3.10
conda activate fmri
pip install numpy pandas nilearn nibabel tqdm matplotlib seaborn

# 2) Configure pipeline
# Edit config.py with your paths, atlases, and QC thresholds

# 3) Check prerequisites
./launcher.sh --check

# 4) Discover subjects
python subject_manager.py --action discover

# 5) Single subject test (includes fMRIPrep + Connectivity + Visualization)
python pipeline_runner.py --mode single --subject ON00400 --cpu-only

# 6) Parallel processing (includes automatic connectivity + visualization)
python pipeline_runner.py --mode parallel --subjects ON00400 ON01016

# 7) Alternative: Standalone connectivity generation
python run_connectivity.py --all-completed

# 8) Run QC
python fmri_qc.py --all

# 9) Monitor status
python pipeline_runner.py --mode status
```

**Note:** Steps 5-6 automatically include connectivity matrix generation and visualization. No separate connectivity command needed!

---

## Command Reference

**Pipeline Runner**
```bash
# Core processing (includes automatic connectivity + visualization)
python pipeline_runner.py --mode {single|parallel|watch|status}
python pipeline_runner.py --mode single --subject SUB_ID [--cpu-only] [--no-connectivity]
python pipeline_runner.py --mode parallel --subjects SUB1 SUB2 [--cpu-only]

# Options
--cpu-only          # Force CPU-only processing
--no-connectivity   # Skip connectivity matrix generation and visualization
--check             # Check prerequisites only
```

**Connectivity Generation**
```bash
# Connectivity matrices + automatic visualization
python connectivity_matrix.py --subjects SUB1 SUB2 --atlases aal schaefer-100
python run_connectivity.py --all-completed --atlases aal harvard_oxford

# All commands automatically generate both CSV matrices AND PNG visualizations
```

**Launcher (Alternative Interface)**
```bash
# Bash interface (includes automatic connectivity + visualization)
./launcher.sh --mode {watch|parallel|status|single}
./launcher.sh --mode single --subject SUB_ID [--cpu-only]
./launcher.sh --check
```

---

## Output Structure
```
OUTPUT_DIR/
├── sub-*/
│   ├── anat/
│   │   ├── *_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
│   │   └── *_desc-brain_mask.nii.gz
│   ├── func/
│   │   ├── *_desc-preproc_bold.nii.gz
│   │   ├── *_desc-confounds_timeseries.tsv
│   │   └── *_desc-carpetplot_bold.svg
│   └── figures/
│       └── *.svg
├── sub-*.html              # fMRIPrep reports
├── logs/
│   ├── sub-*.log
│   ├── sub-*.error
│   └── watcher.log
├── connectivity_matrices/
│   └── sub-*/
│       ├── *_timeseries.csv
│       ├── *_*_connectivity.csv
│       └── *_*_visualization.png
└── qc_reports/
    ├── fmri_qc_summary.html
    └── subjects/
        └── sub-*_ses-*_qc.html
```

---

## Resource Configuration Examples

**High-Performance Single Subject**
```python
# config.py
MAX_PARALLEL = 1
MEMORY_MB = 100000    # 100GB
NPROCS = 24
OMP_NTHREADS = 8
```

**Balanced Parallel Processing**
```python
# config.py
MAX_PARALLEL = 2
MEMORY_MB = 50000     # 50GB per subject
NPROCS = 12
OMP_NTHREADS = 6
```

**CPU-Only Processing**
```python
# config.py
USE_GPU = False
MAX_PARALLEL = 4
MEMORY_MB = 32000     # 32GB per subject
NPROCS = 8
OMP_NTHREADS = 4
```

---

## Troubleshooting

**Common Issues**

*Exit code 126: Permission issues with Singularity*
```bash
chmod +x /path/to/fmriprep_latest.sif
```

*Exit code 2: Subject not found in BIDS directory*
```bash
ls /path/to/bids/directory/sub-*
python subject_manager.py --action list
```

*FreeSurfer license error: Invalid or missing license*
```bash
# Check license file exists and is valid
cat /path/to/license.txt
```

*Out of memory: Reduce resource allocation*
```python
# config.py
MEMORY_MB = 32000  # Reduce from higher values
NPROCS = 8         # Reduce parallel processes
```

*GPU not found: Force CPU-only processing*
```bash
python pipeline_runner.py --mode single --subject SUB_ID --cpu-only
```

---

## fMRIPrep HTML Report Manual Review Guidelines

For detailed quality assessment, review the fMRIPrep HTML reports. Key areas to check:

### **1. Brain Mask & Segmentation**
- **Red line:** Should outline the brain (not skull/dura)
- **Blue line:** Should separate white and grey matter  
- **Check:** No brain tissue cutoff

### **2. Spatial Normalization**
- **Ventricles:** Should be aligned between subject and template
- **White/grey matter boundaries:** Should match
- **Check:** No stretching or distortion

### **3. Fieldmap Correction**
- **Before/after:** Compare images before and after correction
- **Frontal area:** Check distortion correction
- **Check:** Normal brain shape after correction

### **4. Functional-Anatomical Alignment**
- **Red line:** Should align with grey/white matter boundary
- **Fixed/Moving:** No distortion between images
- **Check:** Proper surface registration

### **5. CompCor ROIs**
- **Red line (brain mask):** Should cover full brain
- **Magenta line:** Inside white matter and CSF
- **Blue lines:** In high CSF or blood flow areas

---

## References

- **fMRIPrep:** Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). *fMRIPrep: a robust preprocessing pipeline for functional MRI.* Nature Methods, 16(1), 111–116.  
- **Nilearn:** Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). *Machine learning for neuroimaging with scikit-learn.* Frontiers in Neuroinformatics, 8, 14.  
- **BIDS:** Gorgolewski, K. J., Auer, T., Calhoun, V. D., et al. (2016). *The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments.* Scientific Data, 3, 160044.  
- **QC / Motion Metrics:** Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). *Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion.* NeuroImage, 59(3), 2142–2154.  

---

## Authors / Version
**Author:** Mohammad H. Abbasi (mabbasi@stanford.edu)  
Stanford University, STAI Lab, [https://stai.stanford.edu](https://stai.stanford.edu)  
**Created:** 2025 | **Version:** 1.0.0 | **Last update:** September 16, 2025  

---

## Acknowledgements
Thanks to the **fMRIPrep** team, **Nilearn** contributors, and the broader **neuroimaging community**.  
Special credit to the **STAI Lab** for extending QC, connectivity, and pipeline management.
