# fMRI Processing Pipeline

A lightweight, end-to-end pipeline for fMRI preprocessing using **fMRIPrep** with parallel processing, connectivity matrix generation, CPU/GPU flexibility, and comprehensive quality control.

**Features:** BIDS compatibility → fMRIPrep preprocessing → Connectivity matrices → Parallel processing → Resource monitoring → Queue management.

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
pip install numpy pandas nilearn nibabel tqdm
```

## Configuration

All paths and behavior are controlled by `config.py` (shared by all scripts). The pipeline is config-driven with minimal CLI flags for convenience.

### Key Configuration Items:

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

## Processing Pipeline

### 1) Setup & Prerequisites Check — `launcher.sh --check`

Verifies all required components are available:
- Singularity image
- FreeSurfer license
- Input directory structure
- BIDS filter file

```bash
./launcher.sh --check
```

### 2) Subject Discovery — `subject_manager.py`

Discovers available subjects in BIDS directory and creates processing queue.

**Inputs:** BIDS directory structure (`sub-*/`)
**Outputs:** Subject list file (`remaining_subjects_array.txt`)

```bash
python subject_manager.py --action discover --output subjects.txt
python subject_manager.py --action add --subjects sub-01 sub-02
python subject_manager.py --action remove --subjects sub-01
```

### 3) fMRI Preprocessing — `pipeline_runner.py`

Runs fMRIPrep preprocessing with multiple execution modes:

**Processing Steps:**
1. BIDS validation and input parsing
2. Anatomical preprocessing (skull stripping, registration)
3. Functional preprocessing (motion correction, distortion correction)
4. Surface generation and volume-to-surface mapping
5. Confound estimation and timeseries extraction
6. Output space transformation and resampling

**Outputs (per subject in `OUTPUT_DIR`):**
- `*_desc-preproc_bold.nii.gz`: Preprocessed BOLD timeseries
- `*_desc-confounds_timeseries.tsv`: Confound regressors
- `*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz`: Anatomical in MNI space
- HTML reports with quality metrics

#### Execution Modes:

**Single Subject:**
```bash
python pipeline_runner.py --mode single --subject ON00400 --cpu-only
```

**Parallel Processing:**
```bash
python pipeline_runner.py --mode parallel --subjects ON00400 ON01016 --cpu-only
```

**Watch Mode (Queue Processing):**
```bash
python pipeline_runner.py --mode watch
```

**Status Check:**
```bash
python pipeline_runner.py --mode status
```

### 4) Connectivity Matrix Generation — `connectivity_matrix.py`

Generates functional connectivity matrices using multiple atlases and connectivity measures.

**Processing Steps:**
1. Load preprocessed fMRI data
2. Apply brain atlas parcellation
3. Extract regional timeseries
4. Remove confound signals
5. Compute connectivity matrices (correlation, partial correlation)
6. Save matrices and timeseries as CSV files

**Outputs (per subject in `CONNECTIVITY_OUTPUT_DIR`):**
- `sub-*_timeseries.csv`: Regional timeseries
- `sub-*_*_*_connectivity.csv`: Connectivity matrices

**Available Atlases:**
- `harvard_oxford`: Harvard-Oxford cortical atlas
- `aal`: Automated Anatomical Labeling
- `difumo-256`: DiFuMo 256 regions
- `difumo-1024`: DiFuMo 1024 regions

```bash
python run_connectivity.py --all-completed
python run_connectivity.py --subjects sub-01 sub-02 --atlases aal harvard_oxford
```

### 5) Quality Control — Built-in Reports

fMRIPrep generates comprehensive HTML reports with:
- Motion parameters and framewise displacement
- Registration quality assessment
- Signal-to-noise ratio metrics
- Tissue segmentation overlays
- Functional-anatomical alignment

## Quick Start

```bash
# 1) Setup environment
conda create -y -n fmri python=3.10
conda activate fmri
pip install numpy pandas nilearn nibabel tqdm

# 2) Configure pipeline
# Edit config.py with your paths and settings

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

# 8) Monitor status
python pipeline_runner.py --mode status
```

## Resource Configuration Examples

### High-Performance Single Subject
```python
# config.py
MAX_PARALLEL = 1
MEMORY_MB = 100000    # 100GB
NPROCS = 24
OMP_NTHREADS = 8
```

### Balanced Parallel Processing
```python
# config.py
MAX_PARALLEL = 2
MEMORY_MB = 50000     # 50GB per subject
NPROCS = 12
OMP_NTHREADS = 6
```

### CPU-Only Processing
```python
# config.py
USE_GPU = False
MAX_PARALLEL = 4
MEMORY_MB = 32000     # 32GB per subject
NPROCS = 8
OMP_NTHREADS = 4
```

## Output Structure

```
OUTPUT_DIR/
├── sub-*/
│   ├── anat/
│   │   ├── *_desc-preproc_T1w.nii.gz
│   │   └── *_desc-brain_mask.nii.gz
│   ├── func/
│   │   ├── *_desc-preproc_bold.nii.gz
│   │   ├── *_desc-confounds_timeseries.tsv
│   │   └── *_desc-carpetplot_bold.svg
│   └── figures/
│       └── *.html (QC reports)
├── logs/
│   ├── sub-*.log
│   ├── sub-*.error
│   └── watcher.log
└── connectivity_matrices/
    └── sub-*/
        ├── *_timeseries.csv
        └── *_*_connectivity.csv
```

## Command Reference

### Pipeline Runner
```bash
# Core processing
python pipeline_runner.py --mode {single|parallel|watch|status}
python pipeline_runner.py --mode single --subject SUB_ID [--cpu-only] [--no-connectivity]
python pipeline_runner.py --mode parallel --subjects SUB1 SUB2 [--cpu-only]

# Options
--cpu-only          # Force CPU-only processing
--no-connectivity   # Skip connectivity matrix generation
--check             # Check prerequisites only
```

### Subject Manager
```bash
# Subject management
python subject_manager.py --action {discover|add|remove|list}
python subject_manager.py --action discover [--output FILE]
python subject_manager.py --action add --subjects SUB1 SUB2
python subject_manager.py --action remove --subjects SUB1
```

### Connectivity Generation
```bash
# Connectivity matrices
python run_connectivity.py --all-completed
python run_connectivity.py --subjects SUB1 SUB2
python run_connectivity.py --atlases aal harvard_oxford
python run_connectivity.py --confounds csf white_matter global_signal
python run_connectivity.py --corr_kinds full-corr partial-corr
```

### Launcher (Alternative Interface)
```bash
# Bash interface
./launcher.sh --mode {watch|parallel|status|single}
./launcher.sh --mode single --subject SUB_ID [--cpu-only]
./launcher.sh --check
```

## Troubleshooting

### Common Issues

**Exit code 126:** Permission issues with Singularity
```bash
chmod +x /path/to/fmriprep_latest.sif
```

**Exit code 2:** Subject not found in BIDS directory
```bash
ls /path/to/bids/directory/sub-*
python subject_manager.py --action list
```

**FreeSurfer license error:** Invalid or missing license
```bash
# Check license file exists and is valid
cat /path/to/license.txt
```

**Out of memory:** Reduce resource allocation
```python
# config.py
MEMORY_MB = 32000  # Reduce from higher values
NPROCS = 8         # Reduce parallel processes
```

**GPU not found:** Force CPU-only processing
```bash
python pipeline_runner.py --mode single --subject SUB_ID --cpu-only
```

### Log Analysis
```bash
# Check pipeline status
python pipeline_runner.py --mode status

# View subject logs
tail -f /path/to/logs/sub-SUB_ID.log
cat /path/to/logs/sub-SUB_ID.error

# Monitor system resources
top -u $USER
nvidia-smi  # For GPU usage
```

## References

- **fMRIPrep:** Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. Nature Methods, 16(1), 111-116.
- **Nilearn:** Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). Machine learning for neuroimaging with scikit-learn. Frontiers in Neuroinformatics, 8, 14.
- **BIDS:** Gorgolewski, K. J., Auer, T., Calhoun, V. D., et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Scientific Data, 3, 160044.

## Authors / Version

**Author:** Mohammad H. Abbasi (mabbasi@stanford.edu)  
**Lab:** Stanford University, STAI Lab (https://stai.stanford.edu)  
**Created:** 2025 | **Version:** 1.0.0 | **Last update:** September 15, 2025

## Acknowledgements

Thanks to the fMRIPrep development team, Nilearn contributors, and the broader neuroimaging open-source community for providing robust tools for fMRI analysis.
