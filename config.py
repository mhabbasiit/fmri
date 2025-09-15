#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fMRI Pipeline Configuration
Contains all settings for the fMRI preprocessing pipeline using fMRIPrep

Author: Mohammad Abbasi (mabbasi@stanford.edu)
Based on ADNI-FMRI pipeline configuration
"""

import os

# =============================================================================
# 1. GENERAL CONFIGURATION
# =============================================================================

# Dataset Information
DATASET_NAME = "ADNI fMRI"

# =============================================================================
# 2. PATH CONFIGURATION
# =============================================================================

# Root directory for the project
ROOT_DIR = "/scr/mabbasi/ADNI-FMRI"

# Input/Output Paths
INPUT_DIR = "/simurgh/group/BWM/DataSets/OpenNeuro/raw/ds004215-download/"  # BIDS input directory
OUTPUT_DIR = os.path.join(ROOT_DIR, "FMRI")       # fMRIPrep output directory
WORK_DIR = os.path.join(ROOT_DIR, "work")         # Working directory
LOG_DIR = os.path.join(ROOT_DIR, "logs")          # Log directory

# Subject Management
SUBJECTS_FILE = os.path.join(ROOT_DIR, "remaining_subjects_array.txt")

# =============================================================================
# 3. CONTAINER CONFIGURATION
# =============================================================================

# Singularity/Apptainer Image
SINGULARITY_IMAGE = os.path.join(ROOT_DIR, "fmriprep_latest.sif")

# FreeSurfer License
FREESURFER_LICENSE = os.path.join(ROOT_DIR, "license.txt")
FREESURFER_LICENSE_CONTAINER = "/opt/freesurfer/license.txt"

# BIDS Filter (optional)
BIDS_FILTER_FILE = os.path.join(ROOT_DIR, "bids_filter.json")

# =============================================================================
# 4. HARDWARE CONFIGURATION
# =============================================================================

# GPU Configuration
GPUS = [0, 1, 2, 3]  # Available GPU IDs (4 GPUs for 4 parallel subjects)
USE_GPU = True              # Set to False for CPU-only processing

# Parallel Processing
MAX_PARALLEL = 2  # Maximum parallel processes (2 subjects simultaneously)

# Resource Allocation per Job
MEMORY_MB = 50000      # Memory allocation per job (50GB per subject)
NPROCS = 12            # Number of processes per job (12 per subject)
OMP_NTHREADS = 6       # OpenMP threads per job (6 per subject)

# =============================================================================
# 5. FMRIPREP CONFIGURATION
# =============================================================================

# Output Spaces
OUTPUT_SPACES = ["MNI152NLin2009cAsym:res-2"]

# Processing Options
LONGITUDINAL = True          # Enable longitudinal processing
WRITE_GRAPH = True          # Write workflow graph
SKIP_BIDS_VALIDATION = True # Skip BIDS validation
RESOURCE_MONITOR = True     # Enable resource monitoring
STOP_ON_FIRST_CRASH = True  # Stop on first crash
FS_NO_RECONALL = False      # Use FreeSurfer reconstruction (slower but complete)

# =============================================================================
# 6. WORKFLOW CONFIGURATION
# =============================================================================

# Processing Control
FORCE_REPROCESSING = False   # Force reprocess completed subjects
CLEANUP_WORK = True         # Clean work directory after completion

# Timing Configuration
POLLING_INTERVAL = 10       # Seconds between status checks
LAUNCH_DELAY = 2           # Seconds between subject launches

# =============================================================================
# 7. LOGGING CONFIGURATION
# =============================================================================

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# 8. CONNECTIVITY MATRIX CONFIGURATION
# =============================================================================

# Connectivity Processing
GENERATE_CONNECTIVITY = True    # Generate connectivity matrices after fMRIPrep

# Atlas Configuration
DEFAULT_ATLASES = ['aal', 'schaefer-100']  # Default atlases to use
AVAILABLE_ATLASES = [
    'aal', 'difumo-256', 'difumo-1024', 
    'harvard_oxford', 'schaefer-100', 'schaefer-400'
]

# Confounds Configuration
DEFAULT_CONFOUNDS = [
    "csf", "white_matter", "global_signal", 
    "trans_x", "trans_y", "trans_z", 
    "rot_x", "rot_y", "rot_z"
]

# Connectivity Types
DEFAULT_CONNECTIVITY_TYPES = ['full-corr']
AVAILABLE_CONNECTIVITY_TYPES = ['full-corr', 'partial-corr', 'tangent', 'covariance']

# Connectivity Output
CONNECTIVITY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "connectivity_matrices")

# =============================================================================
# 9. LEGACY COMPATIBILITY
# =============================================================================

# Maintain compatibility with existing scripts
LOGDIR = LOG_DIR
WORKDIR = WORK_DIR
OUTDIR = OUTPUT_DIR
SIF = SINGULARITY_IMAGE
LICENSE_IN_CNT = FREESURFER_LICENSE_CONTAINER
ROOT = ROOT_DIR
LIST = SUBJECTS_FILE

if __name__ == '__main__':
    print("=" * 60)
    print("fMRI Pipeline Configuration")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max Parallel: {MAX_PARALLEL}")
    print(f"GPUs: {GPUS}")
    print("=" * 60)
