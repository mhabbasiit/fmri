# fMRI Processing Pipeline

A lightweight, end-to-end pipeline for fMRI preprocessing using **fMRIPrep** with parallel processing, connectivity matrix generation, CPU/GPU flexibility, and comprehensive quality control.

**Features:**  
BIDS compatibility → fMRIPrep preprocessing → Connectivity matrices → Parallel processing → Resource monitoring → QC evaluation → Queue management.

---

## Prerequisites

- **Python ≥ 3.8**  
- **Singularity/Apptainer** for containerized execution  
- **Python packages:** `numpy`, `pandas`, `nilearn`, `nibabel`, `tqdm`  
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

### 1) Setup & Prerequisites Check — `launcher.sh --check`
```bash
./launcher.sh --check
```

### 2) Subject Discovery — `subject_manager.py`
```bash
python subject_manager.py --action discover --output subjects.txt
python subject_manager.py --action add --subjects sub-01 sub-02
python subject_manager.py --action remove --subjects sub-01
```

### 3) fMRI Preprocessing — `pipeline_runner.py`
- Runs fMRIPrep with BIDS validation, anatomical + functional preprocessing, confounds, and outputs.  

### 4) Connectivity Matrix Generation — `connectivity_matrix.py`
```bash
python run_connectivity.py --all-completed
python run_connectivity.py --subjects sub-01 sub-02 --atlases aal harvard_oxford
```

### 5) Quality Control — `fmri_qc.py`
- Extracts **framewise displacement (FD)** and motion metrics.  
- Computes: Mean FD, Max FD, High-Motion Ratio, Volumes.  
- Labels runs as **PASS / REVIEW / FAIL**.  

```bash
# Run QC for all subjects
python fmri_qc.py --all

# Export summary
python fmri_qc.py --all --export-summary
```

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

# 5) Single subject test
python pipeline_runner.py --mode single --subject ON00400 --cpu-only

# 6) Parallel processing
python pipeline_runner.py --mode parallel --subjects ON00400 ON01016

# 7) Generate connectivity matrices
python run_connectivity.py --all-completed

# 8) Run QC
python fmri_qc.py --all

# 9) Monitor status
python pipeline_runner.py --mode status
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
│       └── *__visualization.png
└── qc_reports/
    ├── fmri_qc_summary.html
    └── subjects/
        └── sub-*_ses-*_qc.html
```

---

## References

- **fMRIPrep:** Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). *fMRIPrep: a robust preprocessing pipeline for functional MRI.* Nature Methods, 16(1), 111–116.  
- **Nilearn:** Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). *Machine learning for neuroimaging with scikit-learn.* Frontiers in Neuroinformatics, 8, 14.  
- **BIDS:** Gorgolewski, K. J., Auer, T., Calhoun, V. D., et al. (2016). *The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments.* Scientific Data, 3, 160044.  
- **QC / Motion Metrics:** Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). *Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion.* NeuroImage, 59(3), 2142–2154.  

---

## Authors / Version
**Author:** Mohammad H. Abbasi (mabbasi@stanford.edu)  
Stanford University, STAI Lab — [https://stai.stanford.edu](https://stai.stanford.edu)  
**Created:** 2025 | **Version:** 1.0.0 | **Last update:** September 16, 2025  

---

## Acknowledgements
Thanks to the **fMRIPrep** team, **Nilearn** contributors, and the broader **neuroimaging community**.  
Special credit to the **STAI Lab** for extending QC, connectivity, and pipeline management.  
