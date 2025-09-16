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
import glob
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
        """Find fMRIPrep output files for a subject (uses config-based structure handling)"""
        subject_dir = os.path.join(OUTPUT_DIR, f"sub-{subject}")
        
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        bold_files = []
        confounds_files = []
        
        if HANDLE_SESSIONS:
            # Check for session-based structure
            sessions = [d for d in os.listdir(subject_dir) 
                       if d.startswith(SESSION_PREFIX) and os.path.isdir(os.path.join(subject_dir, d))]
            
            if sessions:
                # Session-based structure
                self.logger.info(f"Found session-based structure for {subject}: {sessions}")
                for session in sessions:
                    session_func_dir = os.path.join(subject_dir, session, "func")
                    if os.path.exists(session_func_dir):
                        bold_pattern = os.path.join(session_func_dir, f"*{REQUIRED_BOLD_SUFFIX}")
                        confounds_pattern = os.path.join(session_func_dir, f"*{REQUIRED_CONFOUNDS_SUFFIX}")
                        session_bold = glob.glob(bold_pattern)
                        session_confounds = glob.glob(confounds_pattern)
                        
                        # Filter by minimum time points if configured
                        if USE_LARGEST_FILE and MIN_TIME_POINTS > 0:
                            session_bold = self._filter_by_time_points(session_bold, "BOLD", MIN_TIME_POINTS)
                            session_confounds = self._filter_by_time_points(session_confounds, "confounds", MIN_TIME_POINTS)
                        
                        bold_files.extend(session_bold)
                        confounds_files.extend(session_confounds)
                        self.logger.info(f"Session {session}: {len(session_bold)} BOLD files, {len(session_confounds)} confound files")
            else:
                # Non-session-based structure
                self.logger.info(f"Using non-session-based structure for {subject}")
                func_dir = os.path.join(subject_dir, "func")
                if os.path.exists(func_dir):
                    bold_pattern = os.path.join(func_dir, f"*{REQUIRED_BOLD_SUFFIX}")
                    confounds_pattern = os.path.join(func_dir, f"*{REQUIRED_CONFOUNDS_SUFFIX}")
                    bold_files = glob.glob(bold_pattern)
                    confounds_files = glob.glob(confounds_pattern)
        else:
            # Force non-session-based structure
            self.logger.info(f"Using non-session-based structure for {subject} (HANDLE_SESSIONS=False)")
            func_dir = os.path.join(subject_dir, "func")
            if os.path.exists(func_dir):
                bold_pattern = os.path.join(func_dir, f"*{REQUIRED_BOLD_SUFFIX}")
                confounds_pattern = os.path.join(func_dir, f"*{REQUIRED_CONFOUNDS_SUFFIX}")
                bold_files = glob.glob(bold_pattern)
                confounds_files = glob.glob(confounds_pattern)
        
        if not bold_files:
            raise FileNotFoundError(f"No preprocessed BOLD files found for {subject}")
        if not confounds_files:
            raise FileNotFoundError(f"No confounds files found for {subject}")
            
        self.logger.info(f"Found {len(bold_files)} BOLD files and {len(confounds_files)} confound files for {subject}")
        return bold_files, confounds_files

    def _filter_by_time_points(self, file_list, file_type, min_time_points):
        """Filter files by minimum time points requirement"""
        filtered_files = []
        for file_path in file_list:
            try:
                if file_type == "BOLD":
                    img = nib.load(file_path)
                    num_time_points = img.shape[3] if len(img.shape) == 4 else 0
                else:  # confounds
                    confounds = pd.read_csv(file_path, delimiter="\t")
                    num_time_points = len(confounds)
                
                if num_time_points >= min_time_points:
                    filtered_files.append(file_path)
                else:
                    self.logger.warning(f"Skipping {file_path}: {num_time_points} < {min_time_points} time points")
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")
        
        return filtered_files

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

            # Generate visualizations for all connectivity matrices
            self.logger.info(f"Creating visualizations for {subject}")
            viz_success = create_subject_visualizations(subject)
            if viz_success:
                self.logger.info(f"✅ Successfully created visualizations for {subject}")
            else:
                self.logger.warning(f"⚠️ Failed to create some visualizations for {subject}")

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



def create_connectivity_visualization(matrix_file, output_dir, atlas_name, connectivity_type, subject):
    """Create visualization for a connectivity matrix"""
    try:
        # Import matplotlib here to avoid conflicts
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load connectivity matrix
        connectivity_df = pd.read_csv(matrix_file, index_col=0)
        connectivity_matrix = connectivity_df.values
        labels = connectivity_df.columns.tolist()
        
        # Create figure with subplots (1x2 layout)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Connectivity Analysis: {subject} ({atlas_name.upper()} - {connectivity_type})', 
                     fontsize=16, fontweight='bold')
        
        # 1. Full connectivity matrix heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(connectivity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_title('Full Connectivity Matrix')
        ax1.set_xlabel('Brain Regions')
        ax1.set_ylabel('Brain Regions')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Distribution of connectivity values
        ax2 = axes[1]
        # Get lower triangle values (excluding diagonal)
        mask_lower = np.tril(np.ones_like(connectivity_matrix), k=-1).astype(bool)
        connectivity_values = connectivity_matrix[mask_lower]
        
        ax2.hist(connectivity_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(np.mean(connectivity_values), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(connectivity_values):.3f}')
        ax2.axvline(np.median(connectivity_values), color='orange', linestyle='--', 
                    label=f'Median: {np.median(connectivity_values):.3f}')
        ax2.set_title('Distribution of Connectivity Values')
        ax2.set_xlabel('Connectivity Strength')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(connectivity_values):.4f}
Std: {np.std(connectivity_values):.4f}
Strong (>0.5): {np.sum(connectivity_values > 0.5)} ({100*np.sum(connectivity_values > 0.5)/len(connectivity_values):.1f}%)
Weak (<0.1): {np.sum(connectivity_values < 0.1)} ({100*np.sum(connectivity_values < 0.1)/len(connectivity_values):.1f}%)"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(output_dir, f"{subject}_{atlas_name}_{connectivity_type}_visualization.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False

def create_subject_visualizations(subject):
    """Create visualizations for all connectivity matrices of a subject"""
    subject_dir = os.path.join(CONNECTIVITY_OUTPUT_DIR, f"sub-{subject}")
    
    if not os.path.exists(subject_dir):
        print(f"❌ Subject directory not found: {subject_dir}")
        return False
    
    # Find all connectivity matrix files
    matrix_files = [f for f in os.listdir(subject_dir) if f.endswith('_connectivity.csv')]
    
    if not matrix_files:
        print(f"❌ No connectivity matrices found for {subject}")
        return False
    
    print(f"📊 Creating visualizations for {len(matrix_files)} connectivity matrices for {subject}")
    
    success_count = 0
    for matrix_file in matrix_files:
        matrix_path = os.path.join(subject_dir, matrix_file)
        
        # Parse filename to extract atlas and connectivity type
        # Format: sub-{subject}_{atlas}_{connectivity_type}_connectivity.csv
        parts = matrix_file.replace('.csv', '').split('_')
        if len(parts) >= 4:
            atlas_name = parts[-3]
            connectivity_type = parts[-2]
            
            print(f"  🎨 Creating visualization for {atlas_name} - {connectivity_type}")
            
            if create_connectivity_visualization(matrix_path, subject_dir, atlas_name, 
                                               connectivity_type, subject):
                success_count += 1
    
    print(f"✅ Created {success_count}/{len(matrix_files)} visualizations for {subject}")
    return success_count > 0

if __name__ == '__main__':
    main()
