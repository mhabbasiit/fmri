#!/bin/bash
# fMRI Pipeline Launcher Script
# Similar to watch_launch_in_srun.sh but uses configuration from config.py
#
# Usage:
#     ./launcher.sh [--mode watch|parallel|status|single] [--subject SUBJECT] [--check]
#
# Author: Mohammad Abbasi (mabbasi@stanford.edu)
# Based on ADNI-FMRI watch_launch_in_srun.sh

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration (can be overridden by fmri_config.py)
DEFAULT_ROOT="/scr/mabbasi/ADNI-FMRI"
DEFAULT_MAX_PARALLEL=6
DEFAULT_GPUS=(0 1 2 3 4 5)

# Function to get configuration from Python config
get_config() {
    local key=$1
    python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
try:
    from config import $key
    print($key)
except ImportError:
    # Fallback to defaults if config not found
    if '$key' == 'ROOT_DIR': print('$DEFAULT_ROOT')
    elif '$key' == 'MAX_PARALLEL': print('$DEFAULT_MAX_PARALLEL')
    elif '$key' == 'GPUS': print('${DEFAULT_GPUS[*]}')
    else: print('')
except AttributeError:
    print('')
" 2>/dev/null || echo ""
}

# Load configuration
ROOT=$(get_config "ROOT_DIR" || echo "$DEFAULT_ROOT")
MAX_PARALLEL=$(get_config "MAX_PARALLEL" || echo "$DEFAULT_MAX_PARALLEL")
GPUS_STR=$(get_config "GPUS" || echo "${DEFAULT_GPUS[@]}")

# Convert GPUS string to array
if [[ -n "$GPUS_STR" ]]; then
    # Remove brackets and split by space
    GPUS_STR=$(echo "$GPUS_STR" | sed 's/[][]//g' | tr ',' ' ')
    read -ra GPUS <<< "$GPUS_STR"
else
    GPUS=("${DEFAULT_GPUS[@]}")
fi

# Derived paths
LIST="$ROOT/remaining_subjects_array.txt"
LOGDIR="$ROOT/logs"
WORKDIR="$ROOT/work"
OUTDIR="$ROOT/FMRI"
INPUT_DIR=$(get_config "INPUT_DIR" || echo "$ROOT/output_all")
SIF="$ROOT/fmriprep_latest.sif"
LICENSE="$ROOT/license.txt"
LICENSE_IN_CNT="/opt/freesurfer/license.txt"
BIDS_FILTER="$ROOT/bids_filter.json"

# Ensure directories exist
mkdir -p "$LOGDIR"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%F %T')] $1" | tee -a "$LOGDIR/watcher.log"
}

# Function to count running processes
count_running() {
    pgrep -af "fmriprep .* /out participant" | wc -l || echo 0
}

# Function to get next subject from list
get_next_subject() {
    head -1 "$LIST" 2>/dev/null || echo ""
}

# Function to remove first line from list
remove_first_subject() {
    if [[ -f "$LIST" ]]; then
        sed -i '1d' "$LIST"
    fi
}

# Function to run fMRIPrep for a subject
run_fmriprep() {
    local subject=$1
    local gpu=$2
    local cpu_only_flag=${3:-false}
    
    # Set GPU environment (if using GPU)
    local use_gpu=$(get_config "USE_GPU" || echo "true")
    if [[ "$use_gpu" == "True" || "$use_gpu" == "true" ]] && [[ "$cpu_only_flag" != "true" ]]; then
        export CUDA_VISIBLE_DEVICES=$gpu
        local gpu_flag="--nv"
        log_message "Launching $subject using GPU $gpu"
    else
        local gpu_flag=""
        log_message "Launching $subject using CPU-only processing"
    fi
    
    # Set FreeSurfer license environment variable
    export FS_LICENSE="$LICENSE"
    
    # Build bind mounts
    local bind_mounts="-B $INPUT_DIR:/data:ro -B $OUTDIR:/out -B $WORKDIR:/work"
    
    # Check if using FreeSurfer
    local fs_no_reconall=$(get_config "FS_NO_RECONALL" || echo "true")
    if [[ "$fs_no_reconall" != "True" && "$fs_no_reconall" != "true" ]]; then
        bind_mounts="$bind_mounts -B $LICENSE:$LICENSE_IN_CNT:ro"
    fi
    
    # Run fMRIPrep
    singularity run --cleanenv $gpu_flag \
        $bind_mounts \
        "$SIF" /data /out participant \
        --participant-label "$subject" \
        --skip-bids-validation \
        --output-spaces MNI152NLin2009cAsym:res-2 \
        --longitudinal \
        --write-graph \
        --work-dir /work \
        --resource-monitor \
        --stop-on-first-crash \
        --fs-no-reconall \
        --bids-filter-file "$BIDS_FILTER" \
        --omp-nthreads 1 \
        --nprocs 1 \
        --mem-mb 65536 \
        >> "$LOGDIR/${subject}.log" 2>> "$LOGDIR/${subject}.error" &
}

# Function to show status
show_status() {
    local running=$(count_running)
    local remaining=0
    if [[ -f "$LIST" ]]; then
        remaining=$(wc -l < "$LIST")
    fi
    
    echo "=== fMRI Pipeline Status ==="
    echo "Current time: $(date)"
    echo "Running processes: $running"
    echo "Remaining subjects: $remaining"
    echo "Max parallel: $MAX_PARALLEL"
    echo "GPUs: ${GPUS[*]}"
    echo "Root directory: $ROOT"
    echo "Input: $INPUT_DIR"
    echo "Output: $OUTDIR"
    echo "Logs: $LOGDIR"
    
    if [[ -f "$LOGDIR/watcher.log" ]]; then
        echo ""
        echo "Recent activity:"
        tail -5 "$LOGDIR/watcher.log" 2>/dev/null || echo "No recent activity"
    fi
    echo "=========================="
}

# Function to check prerequisites
check_prerequisites() {
    local errors=0
    
    echo "Checking prerequisites..."
    
    # Check singularity image
    if [[ ! -f "$SIF" ]]; then
        echo "❌ Singularity image not found: $SIF"
        errors=$((errors + 1))
    else
        echo "✅ Singularity image found: $SIF"
    fi
    
    # Check license
    if [[ ! -f "$LICENSE" ]]; then
        echo "❌ FreeSurfer license not found: $LICENSE"
        errors=$((errors + 1))
    else
        echo "✅ FreeSurfer license found: $LICENSE"
    fi
    
    # Check input directory
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "❌ Input directory not found: $INPUT_DIR"
        errors=$((errors + 1))
    else
        echo "✅ Input directory found: $INPUT_DIR"
    fi
    
    # Check BIDS filter (optional)
    if [[ -f "$BIDS_FILTER" ]]; then
        echo "✅ BIDS filter found: $BIDS_FILTER"
    else
        echo "⚠️  BIDS filter not found (optional): $BIDS_FILTER"
    fi
    
    # Check subjects list
    if [[ ! -f "$LIST" ]]; then
        echo "❌ Subjects list not found: $LIST"
        errors=$((errors + 1))
    else
        local count=$(wc -l < "$LIST")
        echo "✅ Subjects list found: $LIST ($count subjects)"
    fi
    
    if [[ $errors -gt 0 ]]; then
        echo "❌ $errors error(s) found. Please fix before running."
        return 1
    else
        echo "✅ All prerequisites satisfied"
        return 0
    fi
}

# Function to run in watch mode (like original script)
run_watch_mode() {
    log_message "Starting fMRI pipeline in watch mode (max parallel: $MAX_PARALLEL)"
    
    local idx=0
    
    while true; do
        local running=$(count_running)
        local next_subj=$(get_next_subject)
        
        # Exit if no subjects left and nothing running
        if [[ -z "$next_subj" && "$running" -eq 0 ]]; then
            log_message "All done. Exiting watcher."
            break
        fi
        
        # Launch next subject if slot available
        if [[ "$running" -lt "$MAX_PARALLEL" && -n "$next_subj" ]]; then
            local gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
            
            # Remove subject from list
            remove_first_subject
            
            # Run subject
            run_fmriprep "$next_subj" "$gpu" "$cpu_only"
            
            idx=$((idx + 1))
        fi
        
        sleep 10
    done
}

# Function to use Python runner
run_python_mode() {
    local mode=$1
    echo "Running fMRI pipeline using Python runner in $mode mode..."
    
    if [[ -f "$SCRIPT_DIR/fmri_pipeline_runner.py" ]]; then
        python3 "$SCRIPT_DIR/fmri_pipeline_runner.py" --mode "$mode"
    else
        echo "❌ Python runner not found: $SCRIPT_DIR/fmri_pipeline_runner.py"
        echo "Falling back to watch mode..."
        run_watch_mode
    fi
}

# Main execution
main() {
    local mode="watch"
    local check_only=false
    local single_subject=""
    local gpu_id=0
    local cpu_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                mode="$2"
                shift 2
                ;;
            --check)
                check_only=true
                shift
                ;;
            --subject)
                single_subject="$2"
                shift 2
                ;;
            --gpu)
                gpu_id="$2"
                shift 2
                ;;
            --cpu-only)
                cpu_only=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--mode watch|parallel|status|single] [--subject SUBJECT] [--gpu GPU] [--cpu-only] [--check] [--help]"
                echo ""
                echo "Options:"
                echo "  --mode MODE      Processing mode: watch, parallel, status, or single (default: watch)"
                echo "  --subject SUBJ   Subject ID for single mode (with or without 'sub-' prefix)"
                echo "  --gpu GPU        GPU ID for single subject processing (default: 0)"
                echo "  --cpu-only       Use CPU-only processing (no GPU acceleration)"
                echo "  --check          Check prerequisites only"
                echo "  --help           Show this help message"
                echo ""
                echo "Modes:"
                echo "  watch            Watch mode - continuously process queue (like original script)"
                echo "  parallel         Python parallel mode - process all subjects at once"
                echo "  status           Show current status only"
                echo "  single           Process single subject"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    if [[ "$check_only" == true ]]; then
        exit 0
    fi
    
    # Run based on mode
    case "$mode" in
        status)
            show_status
            ;;
        single)
            if [[ -z "$single_subject" ]]; then
                echo "❌ --subject required for single mode"
                exit 1
            fi
            echo "Processing single subject: $single_subject on GPU ${gpu_id:-0}"
            if [[ -f "$SCRIPT_DIR/pipeline_runner.py" ]]; then
                python3 "$SCRIPT_DIR/pipeline_runner.py" --mode single --subject "$single_subject" --gpu "${gpu_id:-0}"
            else
                echo "❌ Python runner not found, using direct singularity"
                # Ensure subject has 'sub-' prefix
                if [[ ! "$single_subject" =~ ^sub- ]]; then
                    single_subject="sub-$single_subject"
                fi
                run_fmriprep "$single_subject" "${gpu_id:-0}" "$cpu_only"
                wait
            fi
            ;;
        watch)
            run_watch_mode
            ;;
        parallel)
            run_python_mode "parallel"
            ;;
        *)
            echo "❌ Unknown mode: $mode"
            echo "Valid modes: watch, parallel, status, single"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
