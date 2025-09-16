#!/usr/bin/env python3
"""
fMRI Pipeline Runner
Configurable fMRI preprocessing pipeline using fMRIPrep with parallel processing

This script reads all configuration from fmri_config.py and provides:
- Parallel subject processing with GPU allocation
- Automatic subject completion detection
- Work directory cleanup
- Comprehensive logging
- Queue-based subject management

Author: Mohammad Abbasi (mabbasi@stanford.edu)
"""

import os
import sys
import json
import subprocess
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
import glob
from concurrent.futures import ProcessPoolExecutor
import signal

# Import configuration
try:
    from config import *
except ImportError as e:
    print(f"ERROR: Could not import fMRI configuration from config.py")
    print(f"Import error: {e}")
    print("Please ensure config.py is properly configured.")
    sys.exit(1)

class FMRIPipelineRunner:
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.running_processes = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main pipeline log
        log_file = os.path.join(LOG_DIR, f'fmri_pipeline_{timestamp}.log')
        
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger('fmri_pipeline')
        
        # Watcher log for subject tracking
        self.watcher_log = os.path.join(LOG_DIR, 'watcher.log')
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [LOG_DIR, OUTPUT_DIR, WORK_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created/verified directory: {directory}")
            
    def log_watcher(self, message):
        """Log to watcher file with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.watcher_log, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
            
    def check_singularity_image(self):
        """Verify singularity image exists"""
        if not os.path.exists(SINGULARITY_IMAGE):
            self.logger.error(f"Singularity image not found: {SINGULARITY_IMAGE}")
            return False
        return True
        
    def check_freesurfer_license(self):
        """Verify FreeSurfer license exists"""
        if not os.path.exists(FREESURFER_LICENSE):
            self.logger.error(f"FreeSurfer license not found: {FREESURFER_LICENSE}")
            return False
        return True
        
    def has_subject_reports(self, subject):
        """Check if subject has required HTML reports"""
        # Subject-level report
        subject_html = os.path.join(OUTPUT_DIR, f"{subject}.html")
        if os.path.exists(subject_html):
            return True
            
        # Session-level reports (anatomical + functional)
        anat_html = os.path.join(OUTPUT_DIR, f"{subject}_anat.html")
        func_htmls = glob.glob(os.path.join(OUTPUT_DIR, f"{subject}_ses-*_func.html"))
        
        if os.path.exists(anat_html) and func_htmls:
            return True
            
        return False
        
    def log_indicates_success(self, subject):
        """Check if log file indicates successful completion"""
        log_file = os.path.join(LOG_DIR, f"{subject}.log")
        if not os.path.exists(log_file):
            return False
            
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                return any(phrase in content for phrase in [
                    "fMRIPrep finished successfully!",
                    f"✅ Completed {subject}"
                ])
        except Exception:
            return False
            
    def is_subject_completed(self, subject):
        """Check if subject processing is complete (reports + logs)"""
        return self.has_subject_reports(subject) and self.log_indicates_success(subject)
        
    def check_and_run_connectivity(self, subject):
        """Check if connectivity needs to be generated for completed subjects"""
        if not GENERATE_CONNECTIVITY:
            return False
            
        if not self.is_subject_completed(subject):
            return False
            
        # Check if connectivity already exists
        conn_dir = os.path.join(CONNECTIVITY_OUTPUT_DIR, f"sub-{subject}")
        if os.path.exists(conn_dir):
            conn_files = [f for f in os.listdir(conn_dir) if f.endswith('_connectivity.csv')]
            if conn_files:
                self.logger.debug(f"Connectivity matrices already exist for {subject}")
                return False
        
        # Generate connectivity matrices
        self.logger.info(f"🔗 Generating connectivity matrices for completed subject: {subject}")
        success = self.generate_connectivity_matrices(subject)
        
        if success:
            self.logger.info(f"✅ Connectivity generation completed for {subject}")
        else:
            self.logger.error(f"❌ Connectivity generation failed for {subject}")
            
        return success
        
    def cleanup_work_directory(self, subject):
        """Clean up work directory for completed subject"""
        self.logger.info(f"🧹 Cleaning work for {subject}...")
        
        # Direct path under work root
        subj_wf_root = os.path.join(WORK_DIR, f"fmriprep_24_1_wf/{subject}_wf")
        if os.path.exists(subj_wf_root):
            subprocess.run(['rm', '-rf', subj_wf_root], check=False)
            
        # Timestamped roots pattern
        pattern = os.path.join(WORK_DIR, f"*/fmriprep_24_1_wf/{subject}_wf")
        for path in glob.glob(pattern):
            subprocess.run(['rm', '-rf', path], check=False)
            
        self.logger.info(f"✅ Work cleaned for {subject}")
        
    def generate_connectivity_matrices(self, subject):
        """Generate connectivity matrices for a subject"""
        try:
            # Import connectivity processor
            from connectivity_matrix import ConnectivityProcessor
            
            # Initialize processor
            conn_processor = ConnectivityProcessor()
            
            # Fetch atlases
            atlases = conn_processor.fetch_atlases(DEFAULT_ATLASES)
            if not atlases:
                self.logger.error(f"Could not fetch atlases for {subject}")
                return False
                
            # Process connectivity
            success = conn_processor.process_subject_connectivity(
                subject, atlases, DEFAULT_CONFOUNDS, DEFAULT_CONNECTIVITY_TYPES
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Connectivity generation failed for {subject}: {e}")
            return False
        
    def get_subject_list(self):
        """Get list of subjects to process"""
        if os.path.exists(SUBJECTS_FILE):
            with open(SUBJECTS_FILE, 'r') as f:
                subjects = [line.strip() for line in f if line.strip()]
            return subjects
        else:
            self.logger.error(f"Subjects file not found: {SUBJECTS_FILE}")
            return []
            
    def get_remaining_subjects(self):
        """Get list of subjects that still need processing"""
        all_subjects = self.get_subject_list()
        remaining = []
        
        for subject in all_subjects:
            if not self.is_subject_completed(subject):
                remaining.append(subject)
                
        return remaining
        
    def write_remaining_subjects(self, subjects):
        """Write remaining subjects to file"""
        with open(SUBJECTS_FILE, 'w') as f:
            for subject in subjects:
                f.write(f"{subject}\n")
                
    def count_running_processes(self):
        """Count currently running fMRIPrep processes"""
        try:
            result = subprocess.run([
                'pgrep', '-af', 'fmriprep .* /out participant'
            ], capture_output=True, text=True)
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
            return 0
        except Exception:
            return 0
            
    def run_fmriprep_subject(self, subject, gpu_id):
        """Run fMRIPrep for a single subject"""
        log_file = os.path.join(LOG_DIR, f"{subject}.log")
        error_file = os.path.join(LOG_DIR, f"{subject}.error")
        
        # Setup environment
        env = os.environ.copy()
        if USE_GPU:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Set FreeSurfer license environment variable
        env['FS_LICENSE'] = FREESURFER_LICENSE
        
        # Build command
        singularity_args = ['singularity', 'run', '--cleanenv']
        if USE_GPU:
            singularity_args.append('--nv')  # Only add --nv for GPU mode
        
        bind_mounts = [
            '-B', f"{INPUT_DIR}:/data:ro",
            '-B', f"{OUTPUT_DIR}:/out",
            '-B', f"{WORK_DIR}:/work"
        ]
        
        # Only bind FreeSurfer license if using FreeSurfer
        if not FS_NO_RECONALL:
            bind_mounts.extend(['-B', f"{FREESURFER_LICENSE}:{FREESURFER_LICENSE_CONTAINER}:ro"])
        
        cmd = singularity_args + bind_mounts + [
            SINGULARITY_IMAGE,
            '/data', '/out', 'participant',
            '--participant-label', subject,
            '--skip-bids-validation',
            '--output-spaces', ' '.join(OUTPUT_SPACES),
            '--work-dir', '/work',
            '--resource-monitor',
            '--stop-on-first-crash',
            '--omp-nthreads', str(OMP_NTHREADS),
            '--nprocs', str(NPROCS),
            '--mem-mb', str(MEMORY_MB)
        ]
        
        # Add FreeSurfer license only if not using --fs-no-reconall
        if not FS_NO_RECONALL:
            cmd.extend(['--fs-license-file', FREESURFER_LICENSE_CONTAINER])
        
        # Add optional parameters
        if LONGITUDINAL:
            cmd.append('--longitudinal')
        if WRITE_GRAPH:
            cmd.append('--write-graph')
        if FS_NO_RECONALL:
            cmd.append('--fs-no-reconall')
        if BIDS_FILTER_FILE and os.path.exists(BIDS_FILTER_FILE):
            cmd.extend(['--bids-filter-file', BIDS_FILTER_FILE])
            
        # Log start
        if USE_GPU:
            start_msg = f"Starting {subject} on GPU {gpu_id}"
        else:
            start_msg = f"Starting {subject} with CPU-only processing"
        self.logger.info(start_msg)
        self.log_watcher(start_msg)
        
        with open(log_file, 'a') as log_f, open(error_file, 'a') as err_f:
            log_f.write(f"{datetime.now()}: {start_msg}\n")
            
            try:
                # Run fMRIPrep in background (non-blocking)
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=err_f,
                    env=env
                )
                
                # Log process started
                start_success_msg = f"🚀 Launched {subject} (PID: {process.pid})"
                self.logger.info(start_success_msg)
                log_f.write(f"{datetime.now()}: {start_success_msg}\n")
                
                return True
                
            except Exception as e:
                error_msg = f"❌ Failed to launch {subject}: {e}"
                self.logger.error(error_msg)
                self.log_watcher(error_msg)
                err_f.write(f"{datetime.now()}: {error_msg}\n")
                return False
                
    def run_parallel_subjects(self, subjects):
        """Process specific subjects in parallel"""
        self.logger.info(f"Starting parallel processing for {len(subjects)} subjects: {subjects}")
        self.logger.info(f"Max parallel: {MAX_PARALLEL}")
        
        # Limit to MAX_PARALLEL subjects
        subjects_to_process = subjects[:MAX_PARALLEL]
        
        # Launch all subjects immediately (up to MAX_PARALLEL)
        for i, subject in enumerate(subjects_to_process):
            # Assign GPU
            gpu_id = GPUS[i % len(GPUS)] if USE_GPU else 0
            
            self.logger.info(f"Launching subject {i+1}/{len(subjects_to_process)}: {subject}")
            
            # Run subject in background
            self.run_fmriprep_subject(subject, gpu_id)
            
            # Brief pause between launches to avoid conflicts
            time.sleep(3)
            
        self.logger.info(f"All {len(subjects_to_process)} subjects launched successfully")
        
        # Check for connectivity generation after launching
        if GENERATE_CONNECTIVITY:
            self.logger.info("🔗 Checking for completed subjects to generate connectivity matrices...")
            for subject in subjects_to_process:
                self.check_and_run_connectivity(subject)
                
    def run_parallel_processing(self):
        """Run parallel processing with queue management"""
        remaining_subjects = self.get_remaining_subjects()
        total_subjects = len(remaining_subjects)
        
        if total_subjects == 0:
            self.logger.info("🎉 All subjects are already completed!")
            return
            
        self.logger.info(f"Processing {total_subjects} remaining subjects")
        self.logger.info(f"Max parallel processes: {MAX_PARALLEL}")
        
        # Check for already running processes
        running = self.count_running_processes()
        if running > 0:
            self.logger.warning(f"Found {running} running fMRIPrep processes already")
            
        # Process subjects
        completed = 0
        idx = 0
        
        for subject in remaining_subjects:
            # Skip if already completed (double-check)
            if self.is_subject_completed(subject):
                self.logger.info(f"⏭️ Skip {subject} (already completed)")
                continue
                
            # Wait for available slot
            while self.count_running_processes() >= MAX_PARALLEL:
                time.sleep(POLLING_INTERVAL)
                
            # Assign GPU
            gpu_id = GPUS[idx % len(GPUS)]
            
            # Run subject in background
            self.run_fmriprep_subject(subject, gpu_id)
            
            idx += 1
            completed += 1
            
            # Brief pause between launches
            time.sleep(LAUNCH_DELAY)
            
        self.logger.info(f"Launched {completed} subjects")
        
        # After launching all subjects, check for connectivity generation
        if GENERATE_CONNECTIVITY:
            self.logger.info("🔗 Checking for completed subjects to generate connectivity matrices...")
            all_subjects = self.get_subject_list()
            for subject in all_subjects:
                self.check_and_run_connectivity(subject)
        
    def run_watch_mode(self):
        """Run in watch mode - continuously process queue"""
        self.logger.info("Starting watch mode...")
        
        try:
            while True:
                # Check for completed subjects that need connectivity generation
                if GENERATE_CONNECTIVITY:
                    all_subjects = self.get_subject_list()
                    for subject in all_subjects:
                        self.check_and_run_connectivity(subject)
                
                # Get next subject from file
                subjects = self.get_subject_list()
                if not subjects:
                    running = self.count_running_processes()
                    if running == 0:
                        self.logger.info("All done. Exiting watcher.")
                        break
                    else:
                        self.logger.info(f"No subjects in queue, but {running} still running")
                        time.sleep(POLLING_INTERVAL)
                        continue
                        
                next_subject = subjects[0]
                running = self.count_running_processes()
                
                if running < MAX_PARALLEL:
                    # Remove subject from file
                    remaining_subjects = subjects[1:]
                    self.write_remaining_subjects(remaining_subjects)
                    
                    # Assign GPU
                    gpu_id = GPUS[len(self.running_processes) % len(GPUS)]
                    
                    # Launch subject
                    self.run_fmriprep_subject(next_subject, gpu_id)
                    
                time.sleep(POLLING_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal. Stopping watcher...")
            
    def generate_status_report(self):
        """Generate current processing status"""
        all_subjects = self.get_subject_list()
        remaining = self.get_remaining_subjects()
        completed = len(all_subjects) - len(remaining)
        running = self.count_running_processes()
        
        print(f"\n=== fMRI Pipeline Status ===")
        print(f"Total subjects: {len(all_subjects)}")
        print(f"Completed: {completed}")
        print(f"Remaining: {len(remaining)}")
        print(f"Currently running: {running}")
        print(f"Max parallel: {MAX_PARALLEL}")
        print(f"GPUs: {GPUS}")
        print(f"Input: {INPUT_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Logs: {LOG_DIR}")
        
        # Show recent watcher activity
        if os.path.exists(self.watcher_log):
            print(f"\nRecent activity:")
            subprocess.run(['tail', '-5', self.watcher_log])

def main():
    parser = argparse.ArgumentParser(description='fMRI Pipeline Runner')
    parser.add_argument('--mode', choices=['parallel', 'watch', 'status', 'single'], 
                       default='watch',
                       help='Processing mode (default: watch)')
    parser.add_argument('--subject', 
                       help='Process single subject (use with --mode single)')
    parser.add_argument('--subjects', nargs='+',
                       help='Process multiple subjects (use with --mode parallel)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID for single subject processing (default: 0)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Use CPU-only processing (no GPU acceleration)')
    parser.add_argument('--no-connectivity', action='store_true',
                       help='Skip connectivity matrix generation')
    parser.add_argument('--check', action='store_true',
                       help='Check prerequisites only')
    
    args = parser.parse_args()
    
    # Override USE_GPU setting if --cpu-only specified
    if args.cpu_only:
        global USE_GPU
        USE_GPU = False
    
    # Override connectivity setting if --no-connectivity specified
    if args.no_connectivity:
        global GENERATE_CONNECTIVITY
        GENERATE_CONNECTIVITY = False
    
    runner = FMRIPipelineRunner()
    
    # Check prerequisites
    if not runner.check_singularity_image():
        sys.exit(1)
    if not runner.check_freesurfer_license():
        sys.exit(1)
        
    if args.check:
        runner.logger.info("✅ All prerequisites check passed")
        return
        
    if args.mode == 'status':
        runner.generate_status_report()
    elif args.mode == 'single':
        if not args.subject:
            print("ERROR: --subject required for single mode")
            sys.exit(1)
        # Ensure subject has 'sub-' prefix
        subject = args.subject if args.subject.startswith('sub-') else f"sub-{args.subject}"
        success = runner.run_fmriprep_subject(subject, args.gpu)
        sys.exit(0 if success else 1)
    elif args.mode == 'parallel':
        if args.subjects:
            # Process specific subjects
            subjects = [s if s.startswith('sub-') else f"sub-{s}" for s in args.subjects]
            runner.run_parallel_subjects(subjects)
        else:
            # Process from queue file
            runner.run_parallel_processing()
    elif args.mode == 'watch':
        runner.run_watch_mode()

if __name__ == '__main__':
    main()
