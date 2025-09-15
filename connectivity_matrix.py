#!/usr/bin/env python3
"""
fMRI Connectivity Matrix Generation
Post-processing fMRI images with Nilearn to obtain connectivity matrices

Author: Mohammad Hassan Abbasi (based on Favour Nerrise's work)
Date: September 15, 2025

Purpose:
    This module processes pre-processed images from fMRIPrep and generates connectivity
    matrices based on specified atlas parcellation. Integrated with fMRI pipeline config.
"""

import os
import sys
import numpy as np
import pandas as pd
from nilearn import datasets, input_data, connectome
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
import argparse
import nibabel as nib
import logging
from datetime import datetime
from pathlib import Path
import glob

# Import configuration
try:
    from config import *
except ImportError as e:
    print(f"ERROR: Could not import configuration from config.py: {e}")
    sys.exit(1)

# Set up logging
def setup_logging():
    """Setup logging for connectivity processing"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_file = os.path.join(LOG_DIR, f'connectivity_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger('connectivity')

class ConnectivityProcessor:
    def __init__(self):
        self.logger = setup_logging()
        self.connectivity_dir = os.path.join(OUTPUT_DIR, "connectivity_matrices")
        os.makedirs(self.connectivity_dir, exist_ok=True)
        
    def fetch_atlases(self, atlas_names):
        """Fetches specified atlases from Nilearn datasets"""
        atlases = {}
        for atlas_name in atlas_names:
            self.logger.info(f"Fetching data for atlas: {atlas_name}")
            try:
                if atlas_name == "harvard_oxford":
                    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
                    atlases[atlas_name] = (atlas.maps, atlas.labels[1:])
                elif atlas_name == "difumo-256":
                    atlas = datasets.fetch_atlas_difumo(dimension=256, resolution_mm=2, legacy_format=False)
                    atlases[atlas_name] = (atlas.maps, atlas.labels)
                elif atlas_name == "difumo-1024":
                    atlas = datasets.fetch_atlas_difumo(dimension=1024, resolution_mm=2, legacy_format=False)
                    atlases[atlas_name] = (atlas.maps, atlas.labels)
                elif atlas_name == "aal":
                    atlas = datasets.fetch_atlas_aal()
                    atlases[atlas_name] = (atlas.maps, atlas.labels)
                elif atlas_name == "schaefer-100":
                    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
                    atlases[atlas_name] = (atlas.maps, atlas.labels)
                elif atlas_name == "schaefer-400":
                    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
                    atlases[atlas_name] = (atlas.maps, atlas.labels)
                else:
                    raise ValueError(f"Unsupported atlas: {atlas_name}")
                    
                self.logger.info(f"Successfully fetched {atlas_name}")
            except Exception as e:
                self.logger.error(f"Failed to fetch {atlas_name}: {e}")
                continue
                
        return atlases

    def _get_correlation(self, kind):
        """Convert correlation kind to nilearn format"""
        correlation_map = {
            'full-corr': 'correlation',
            'partial-corr': 'partial correlation',
            'tangent': 'tangent',
            'covariance': 'covariance'
        }
        
        if kind not in correlation_map:
            raise ValueError(f"Unsupported correlation kind: {kind}")
        return correlation_map[kind]

    def _get_masker(self, atlas_name, atlas_map):
        """Get appropriate masker for atlas type"""
        if 'difumo' in atlas_name:
            masker = NiftiMapsMasker(
                maps_img=atlas_map, 
                standardize="zscore",
                memory_level=1,
                verbose=0
            )
        else:
            masker = NiftiLabelsMasker(
                labels_img=atlas_map, 
                standardize=True,
                memory_level=1,
                verbose=0
            )
        
        return masker

    def find_fmriprep_files(self, subject):
        """Find fMRIPrep output files for a subject"""
        subject_dir = os.path.join(OUTPUT_DIR, f"sub-{subject}")
        
        # Find preprocessed BOLD files
        bold_pattern = os.path.join(subject_dir, "func", "*_desc-preproc_bold.nii.gz")
        bold_files = glob.glob(bold_pattern)
        
        # Find confounds files
        confounds_pattern = os.path.join(subject_dir, "func", "*_desc-confounds_timeseries.tsv")
        confounds_files = glob.glob(confounds_pattern)
        
        if not bold_files:
            raise FileNotFoundError(f"No preprocessed BOLD files found for {subject}")
        if not confounds_files:
            raise FileNotFoundError(f"No confounds files found for {subject}")
            
        return bold_files, confounds_files

    def _get_largest_file(self, file_list, file_type="BOLD"):
        """Get file with most time points"""
        largest_file = None
        largest_size = 0
        
        for file_path in file_list:
            try:
                if file_type == "BOLD":
                    img = nib.load(file_path)
                    num_time_points = img.shape[3] if len(img.shape) == 4 else 0
                else:  # Confounds
                    confounds = pd.read_csv(file_path, delimiter="\t")
                    num_time_points = len(confounds)
                    
                if num_time_points > largest_size:
                    largest_size = num_time_points
                    largest_file = file_path
                    
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")
                continue
                
        if largest_file:
            self.logger.info(f"Selected {file_type} file with {largest_size} time points: {os.path.basename(largest_file)}")
            return largest_file
        else:
            raise Exception(f"No suitable {file_type} file found")

    def process_subject_connectivity(self, subject, atlases, confounds_list, corr_kinds):
        """Process connectivity matrices for a single subject"""
        self.logger.info(f"Processing connectivity for subject {subject}")
        
        try:
            # Find fMRIPrep output files
            bold_files, confounds_files = self.find_fmriprep_files(subject)
            
            # Select files with most time points
            fmri_file = self._get_largest_file(bold_files, "BOLD")
            confounds_file = self._get_largest_file(confounds_files, "confounds")
            
            # Create subject output directory
            subject_conn_dir = os.path.join(self.connectivity_dir, f"sub-{subject}")
            os.makedirs(subject_conn_dir, exist_ok=True)
            
            # Load confounds
            confounds_df = pd.read_csv(confounds_file, delimiter="\t")
            self.logger.info(f"Loaded confounds: {confounds_df.shape}")
            
            # Check if all requested confounds exist
            available_confounds = [col for col in confounds_list if col in confounds_df.columns]
            missing_confounds = [col for col in confounds_list if col not in confounds_df.columns]
            
            if missing_confounds:
                self.logger.warning(f"Missing confounds for {subject}: {missing_confounds}")
                
            if not available_confounds:
                self.logger.error(f"No valid confounds found for {subject}")
                return False
                
            confounds_array = confounds_df[available_confounds].fillna(0).to_numpy()
            
            # Process each atlas
            for atlas_name, (atlas_map, labels) in atlases.items():
                self.logger.info(f"Processing {subject} with {atlas_name} atlas")
                
                try:
                    # Create masker and extract time series
                    masker = self._get_masker(atlas_name, atlas_map)
                    time_series = masker.fit_transform(fmri_file, confounds=confounds_array)
                    
                    self.logger.info(f"Extracted time series: {time_series.shape}")
                    
                    # Save time series
                    timeseries_file = os.path.join(subject_conn_dir, f"sub-{subject}_{atlas_name}_timeseries.csv")
                    pd.DataFrame(time_series).to_csv(timeseries_file, index=False)
                    self.logger.info(f"Saved time series: {timeseries_file}")
                    
                    # Generate connectivity matrices
                    for corr_kind in corr_kinds:
                        try:
                            self.logger.info(f"Computing {corr_kind} connectivity")
                            corr_method = self._get_correlation(corr_kind)
                            
                            correlation_measure = connectome.ConnectivityMeasure(kind=corr_method)
                            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                            
                            # Save connectivity matrix
                            matrix_file = os.path.join(subject_conn_dir, 
                                                     f"sub-{subject}_{atlas_name}_{corr_kind}_connectivity.csv")
                            
                            # Create DataFrame with labels if available
                            if len(labels) == correlation_matrix.shape[0]:
                                matrix_df = pd.DataFrame(correlation_matrix, index=labels, columns=labels)
                            else:
                                matrix_df = pd.DataFrame(correlation_matrix)
                                
                            matrix_df.to_csv(matrix_file)
                            self.logger.info(f"Saved connectivity matrix: {matrix_file}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to compute {corr_kind} for {subject}/{atlas_name}: {e}")
                            continue
                            
                except Exception as e:
                    self.logger.error(f"Failed to process {atlas_name} for {subject}: {e}")
                    continue
                    
            self.logger.info(f"Completed connectivity processing for {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process connectivity for {subject}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Generate connectivity matrices from fMRIPrep outputs')
    parser.add_argument('--subjects', nargs='+', help='List of subject IDs to process')
    parser.add_argument('--subjects-file', help='File containing subject IDs (one per line)')
    parser.add_argument('--atlases', nargs='+', 
                       default=['aal', 'schaefer-100'], 
                       choices=['aal', 'difumo-256', 'difumo-1024', 'harvard_oxford', 'schaefer-100', 'schaefer-400'],
                       help='Atlas names for parcellation')
    parser.add_argument('--confounds', nargs='+', 
                       default=["csf", "white_matter", "global_signal", "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
                       help='Confound regressors to remove')
    parser.add_argument('--corr-kinds', nargs='+', 
                       default=['full-corr'],
                       choices=['full-corr', 'partial-corr', 'tangent', 'covariance'],
                       help='Types of connectivity matrices')
    parser.add_argument('--parallel', action='store_true',
                       help='Process subjects in parallel')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ConnectivityProcessor()
    
    # Get subject list
    if args.subjects:
        subjects = [s.replace('sub-', '') for s in args.subjects]  # Remove sub- prefix if present
    elif args.subjects_file:
        with open(args.subjects_file, 'r') as f:
            subjects = [line.strip().replace('sub-', '') for line in f if line.strip()]
    else:
        # Use subjects from config file
        with open(SUBJECTS_FILE, 'r') as f:
            subjects = [line.strip().replace('sub-', '') for line in f if line.strip()]
    
    processor.logger.info(f"Processing {len(subjects)} subjects")
    
    # Fetch atlases
    atlases = processor.fetch_atlases(args.atlases)
    if not atlases:
        processor.logger.error("No atlases could be fetched")
        sys.exit(1)
    
    # Process subjects
    successful = 0
    failed = 0
    
    if args.parallel:
        # TODO: Implement parallel processing
        processor.logger.info("Parallel processing not yet implemented, using sequential")
    
    for subject in subjects:
        processor.logger.info(f"Processing subject {subject}")
        try:
            success = processor.process_subject_connectivity(subject, atlases, args.confounds, args.corr_kinds)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            processor.logger.error(f"Error processing {subject}: {e}")
            failed += 1
    
    processor.logger.info(f"Connectivity processing completed: {successful} successful, {failed} failed")

if __name__ == '__main__':
    main()

