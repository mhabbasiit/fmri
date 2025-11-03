#!/usr/bin/env python3
"""
fMRI Parcellation Helper (Special Use Case)
Configurable helper for post-processing preprocessed fMRIPrep outputs with:
- Multi-session handling (pick best run by timepoints)
- Optional fast local I/O staging for large files
- Connectivity matrix generation using multiple atlases

This script is intended for dataset-specific or experimental runs that
need manual control beyond the main pipeline runner.

Author: Mohammad Hassan Abbasi (mabbasi@stanford.edu), Favour Nerrise (fnerrise@stanford.edu)
Updated by: Karan Singth (karanps@stanford.edu)
Last updated: November 2025

Example:
  python scripts/parcellation_helper_special.py \
      --atlases difumo-1024 \
      --subjects_file subjects.txt \
      --root_dir /path/to/bids/derivatives/ \
      --save_dir /path/to/connectivity/ \
      --use-temp-storage \
      --temp-dir /tmp/fmri_fastio
"""

import os
import time
import argparse
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, connectome
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker


# ------------------------------
# Fast I/O Utilities (generic)
# ------------------------------
def copy_to_temp_storage(fmri_file: str,
                         confounds_file: str | None,
                         temp_dir: str = "/tmp/fmri_fastio") -> tuple[str, str | None]:
    """Copy fMRI and confounds files to local temp storage for faster I/O."""
    os.makedirs(temp_dir, exist_ok=True)

    # fMRI
    t0 = time.time()
    fmri_temp = os.path.join(temp_dir, os.path.basename(fmri_file))
    shutil.copy2(fmri_file, fmri_temp)
    dt = time.time() - t0
    size_mb = os.path.getsize(fmri_file) / (1024 * 1024)
    speed = size_mb / dt if dt > 0 else float("inf")
    print(f"\tStaged fMRI ({size_mb:.1f} MB) in {dt:.2f}s ~ {speed:.1f} MB/s")

    # Confounds
    conf_temp = None
    if confounds_file is not None:
        t0 = time.time()
        conf_temp = os.path.join(temp_dir, os.path.basename(confounds_file))
        shutil.copy2(confounds_file, conf_temp)
        print(f"\tStaged confounds in {time.time() - t0:.2f}s")

    return fmri_temp, conf_temp


def cleanup_temp_files(*files: str) -> None:
    """Remove temporary files after processing."""
    for f in files:
        try:
            if f and os.path.exists(f):
                os.remove(f)
                print(f"\tCleaned temp: {f}")
        except Exception as e:
            print(f"\tWarning: could not delete {f}: {e}")


# ------------------------------
# Atlas / Masker Utilities
# ------------------------------
def fetch_atlases(atlas_names: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Download/load requested atlases via nilearn."""
    atlases: dict[str, tuple[str, list[str]]] = {}
    for atlas_name in atlas_names:
        print(f"Fetching atlas: {atlas_name}")
        if atlas_name == "harvard_oxford":
            atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
            atlases[atlas_name] = (atlas.maps, atlas.labels[1:])  # drop background
        elif atlas_name == "difumo-128":
            atlas = datasets.fetch_atlas_difumo(dimension=128, resolution_mm=2, legacy_format=False)
            atlases[atlas_name] = (atlas.maps, atlas.labels)
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
    return atlases


def get_masker(atlas_name: str, atlas_map: str):
    """Return appropriate nilearn masker for atlas type."""
    if "difumo" in atlas_name or "schaefer" in atlas_name:
        return NiftiMapsMasker(maps_img=atlas_map)  # can add standardize="zscore" if desired
    return NiftiLabelsMasker(labels_img=atlas_map, standardize=True)


def kind_to_nilearn(kind: str) -> str:
    if kind == "full-corr":
        return "correlation"
    if kind == "partial-corr":
        return "partial correlation"
    if kind == "tangent":
        return "tangent"
    if kind == "covariance":
        return "covariance"
    raise ValueError(f"Unsupported connectivity kind: {kind}")


# ------------------------------
# File Discovery Helpers (generic)
# ------------------------------
def list_sessions(root_dir: str, subject: str) -> list[str]:
    """Return available sessions for a subject (expects sub-*/ses-*/func layout)."""
    subj_dir = os.path.join(root_dir, f"sub-{subject}")
    if not os.path.isdir(subj_dir):
        return []
    sessions = [
        d for d in os.listdir(subj_dir)
        if d.startswith("ses-") and os.path.isdir(os.path.join(subj_dir, d, "func"))
    ]
    return sorted(sessions)


def _largest_bold(func_dir: str) -> str | None:
    """Pick the preprocessed BOLD with the largest time dimension."""
    best_path, best_tp = None, -1
    for f in os.listdir(func_dir):
        if f.endswith("_desc-preproc_bold.nii.gz"):
            p = os.path.join(func_dir, f)
            try:
                n_tp = nib.load(p).shape[3]
                if n_tp > best_tp:
                    best_tp = n_tp
                    best_path = p
            except Exception:
                continue
    return best_path


def _longest_confounds(func_dir: str) -> str | None:
    """Pick the confounds TSV with the most rows (fallback/robust selection)."""
    best_path, best_rows = None, -1
    for f in os.listdir(func_dir):
        if f.endswith("_desc-confounds_timeseries.tsv"):
            p = os.path.join(func_dir, f)
            try:
                # fast-ish row count
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    n = sum(1 for _ in fh) - 1
                if n > best_rows:
                    best_rows = n
                    best_path = p
            except Exception:
                continue
    return best_path


def find_files(root_dir: str, subject: str, session: str) -> tuple[str | None, str | None]:
    """Return (bold_path, confounds_path) for a subject/session."""
    func_dir = os.path.join(root_dir, f"sub-{subject}", session, "func")
    if not os.path.isdir(func_dir):
        return None, None
    return _largest_bold(func_dir), _longest_confounds(func_dir)


# ------------------------------
# Subject Processing
# ------------------------------
def process_subject(subject: str,
                    atlas_name: str,
                    atlas_map: str,
                    labels: list[str],
                    confounds_list: list[str],
                    corr_kinds: list[str],
                    root_dir: str,
                    save_dir: str,
                    use_temp_storage: bool = False,
                    temp_dir: str = "/tmp/fmri_fastio",
                    min_timepoints: int = 1) -> None:
    sessions = list_sessions(root_dir, subject)
    if not sessions:
        print(f"No sessions found for {subject}")
        return

    for ses in sessions:
        print(f"Processing {subject} - {ses} ({atlas_name})")

        bold_path, conf_path = find_files(root_dir, subject, ses)
        if not bold_path or not conf_path:
            print(f"\tSkipping {subject} {ses}: missing BOLD/confounds")
            continue

        # Load confounds first to verify length & columns
        conf_df = pd.read_csv(conf_path, sep="\t")
        conf_cols = [c for c in confounds_list if c in conf_df.columns]
        if not conf_cols:
            print(f"\tNo requested confounds available in {conf_path}; skipping.")
            continue

        # Verify time points
        try:
            n_tp_bold = nib.load(bold_path).shape[3]
        except Exception as e:
            print(f"\tCould not read BOLD: {e}; skipping.")
            continue

        n_tp_conf = len(conf_df)
        if n_tp_bold < min_timepoints:
            print(f"\tToo few timepoints ({n_tp_bold} < {min_timepoints}); skipping.")
            continue

        # Allow 1 TR mismatch (e.g., initial volumes dropped)
        if abs(n_tp_bold - n_tp_conf) > 1:
            print(f"\tMismatch TRs (bold={n_tp_bold}, conf={n_tp_conf}); skipping.")
            continue

        # Optionally stage to local temp (faster I/O)
        staged_bold, staged_conf = None, None
        if use_temp_storage:
            staged_bold, staged_conf = copy_to_temp_storage(bold_path, conf_path, temp_dir=temp_dir)
            bold_for_masker = staged_bold
            conf_for_masker = staged_conf
        else:
            bold_for_masker = bold_path
            conf_for_masker = conf_path

        try:
            masker = get_masker(atlas_name, atlas_map)
            conf_arr = conf_df[conf_cols].fillna(0).to_numpy()

            # Extract ROI time series
            ts = masker.fit_transform(bold_for_masker, confounds=conf_arr)

            # Save timeseries
            os.makedirs(save_dir, exist_ok=True)
            base = f"sub-{subject}_{ses}_{atlas_name}"
            ts_out = os.path.join(save_dir, f"{base}_timeseries.csv")
            pd.DataFrame(ts).to_csv(ts_out, index=False)
            print(f"\tSaved {ts_out}")

            # Connectivity matrices
            for k in corr_kinds:
                nk = kind_to_nilearn(k)
                cm = connectome.ConnectivityMeasure(kind=nk, vectorize=False, discard_diagonal=False)
                mat = cm.fit_transform([ts])[0]
                if labels and len(labels) == mat.shape[0]:
                    df = pd.DataFrame(mat, index=labels, columns=labels)
                else:
                    df = pd.DataFrame(mat)
                out = os.path.join(save_dir, f"{base}_{k}_connectivity.csv")
                df.to_csv(out)
                print(f"\tSaved {out}")

        finally:
            if use_temp_storage:
                cleanup_temp_files(staged_bold, staged_conf)


# ------------------------------
# CLI
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="Special-use fMRI parcellation helper.")
    p.add_argument("--atlases", nargs="+", default=["difumo-1024"],
                   help="One or more atlases (e.g., aal harvard_oxford difumo-1024 schaefer-400)")
    p.add_argument("--subjects_file", required=True,
                   help="Path to a text file containing subject IDs (one per line, without 'sub-').")
    p.add_argument("--root_dir", required=True,
                   help="Path to derivatives root containing sub-*/ses-*/func/*_desc-preproc_bold.nii.gz")
    p.add_argument("--save_dir", required=True,
                   help="Directory to save timeseries and connectivity CSVs.")
    p.add_argument("--confounds", nargs="+",
                   default=["csf", "white_matter", "global_signal",
                            "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
                   help="Confound columns to use from fMRIPrep confounds TSV.")
    p.add_argument("--corr_kinds", nargs="+", default=["full-corr"],
                   choices=["full-corr", "partial-corr", "tangent", "covariance"],
                   help="Connectivity matrix types to compute.")
    p.add_argument("--use-temp-storage", action="store_true",
                   help="Stage large files to a fast local directory for I/O speedup.")
    p.add_argument("--temp-dir", default="/tmp/fmri_fastio",
                   help="Temp directory for staged files (used with --use-temp-storage).")
    p.add_argument("--min-timepoints", type=int, default=1,
                   help="Minimum TRs required to process a run.")
    args = p.parse_args()

    atlases = fetch_atlases(args.atlases)
    with open(args.subjects_file, "r") as f:
        subjects = [line.strip() for line in f if line.strip()]

    for sub in subjects:
        for atlas_name, (atlas_map, labels) in atlases.items():
            process_subject(
                subject=sub,
                atlas_name=atlas_name,
                atlas_map=atlas_map,
                labels=labels,
                confounds_list=args.confounds,
                corr_kinds=args.corr_kinds,
                root_dir=args.root_dir,
                save_dir=args.save_dir,
                use_temp_storage=args.use_temp_storage,
                temp_dir=args.temp_dir,
                min_timepoints=args.min_timepoints,
            )


if __name__ == "__main__":
    main()
