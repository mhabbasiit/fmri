# fMRI Pipeline - Quick Usage Guide

## 🚀 Quick Start (5 Steps)

### 1. Configure Pipeline
```bash
# Edit config.py with your paths
INPUT_DIR = "/path/to/your/bids/data"
OUTPUT_DIR = "/path/to/output"
```

### 2. Check Prerequisites
```bash
./launcher.sh --check
```

### 3. Process Single Subject
```bash
python pipeline_runner.py --mode single --subject ON00400 --cpu-only
```

### 4. Process Multiple Subjects in Parallel
```bash
python pipeline_runner.py --mode parallel --subjects ON00400 ON01016 --cpu-only
```

### 5. Generate Connectivity Matrices
```bash
python run_connectivity.py --all-completed
```

## 📊 Monitor Progress

```bash
# Check current status
python pipeline_runner.py --mode status

# View logs
tail -f ADNI-FMRI/logs/sub-*.log
```

## ⚙️ Common Commands

| Task | Command |
|------|---------|
| **Single subject** | `python pipeline_runner.py --mode single --subject SUB_ID` |
| **Parallel (2 subjects)** | `python pipeline_runner.py --mode parallel --subjects SUB1 SUB2` |
| **CPU-only processing** | Add `--cpu-only` to any command |
| **Skip connectivity** | Add `--no-connectivity` to any command |
| **Check status** | `python pipeline_runner.py --mode status` |
| **List subjects** | `python subject_manager.py --action list` |
| **Add subjects to queue** | `python subject_manager.py --action add --subjects SUB1 SUB2` |

## 🔧 Resource Configuration

**High-performance single subject:**
```python
MAX_PARALLEL = 1
MEMORY_MB = 100000  # 100GB
NPROCS = 24
OMP_NTHREADS = 8
```

**Balanced parallel processing:**
```python
MAX_PARALLEL = 2
MEMORY_MB = 50000   # 50GB per subject
NPROCS = 12
OMP_NTHREADS = 6
```

## 📁 Key Files

- `config.py` - Main configuration
- `pipeline_runner.py` - Core processing script
- `launcher.sh` - Bash interface (alternative)
- `subject_manager.py` - Subject queue management
- `connectivity_matrix.py` - Connectivity analysis
- `run_connectivity.py` - Standalone connectivity script

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Subject not found"** | Check BIDS directory: `ls /path/to/bids/sub-*` |
| **"FreeSurfer license error"** | Verify license file exists and is valid |
| **"Exit code 126"** | Check Singularity permissions |
| **Out of memory** | Reduce `MEMORY_MB` in config.py |
| **No GPU found** | Use `--cpu-only` flag |

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. View logs in `ADNI-FMRI/logs/`
3. Run `python pipeline_runner.py --mode status` for current state
4. Contact: mabbasi@stanford.edu

---
**Quick tip:** Always start with a single subject test before running parallel processing!
