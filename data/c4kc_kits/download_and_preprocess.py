"""
C4KC-KiTS Download + Preprocessing Pipeline
============================================
Downloads kidney CT data from TCIA and preprocesses into 2D PNG images
for deep learning survival analysis.

Dataset: C4KC-KiTS (210 patients, arterial-phase CT + kidney tumor segmentation)
Source: https://www.cancerimagingarchive.net/collection/c4kc-kits/

Preprocessing Steps (based on published KiTS standards):
  1. Download clinical CSV + DICOM CTs + segmentations from TCIA REST API
  2. Convert DICOM → NIfTI via dcm2niix (CT) + dcmqi segimage2itkimage (SEG)
  3. Isotropic resampling to 1×1×1 mm³ (scipy ndimage)
  4. Abdominal soft tissue windowing: clip [-160, 240] HU (W=400, L=40)
  5. Slice selection: axial slice with largest kidney tumor cross-section
  6. Resize to 224×224, save as grayscale PNG

References:
  - Heller et al. (2021) KiTS19 challenge, Med Image Anal 67, 101821
  - RadiomicsHub/C4KC_KiTS preprocessing pipeline
  - Standardized abdominal CT windowing for renal mass characterization

Usage on HPC:
  python data/c4kc_kits/download_and_preprocess.py --out_dir $SCRATCH/Q-Surv/data/c4kc_kits
"""

import os
import sys
import json
import zipfile
import argparse
import requests
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy.ndimage import zoom as scipy_zoom
except ImportError:
    scipy_zoom = None

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
CLINICAL_CSV_URL = (
    "https://wiki.cancerimagingarchive.net/download/attachments/61081171/"
    "C4KC%20KiTS_Clinical%20Data_Version%201.csv?api=v2"
)
COLLECTION = "C4KC-KiTS"

# Abdominal soft tissue window (standard for kidney/renal tumors)
HU_MIN = -160   # W=400, L=40 → lower = L - W/2 = 40 - 200 = -160
HU_MAX = 240    # upper = L + W/2 = 40 + 200 = 240
TARGET_SPACING = (1.0, 1.0, 1.0)
IMG_SIZE = 224


import time

def tcia_get(endpoint, params=None, max_retries=5):
    """Query TCIA REST API with exponential backoff for rate limits."""
    url = f"{TCIA_BASE}/{endpoint}"
    if params is None:
        params = {}
    params["format"] = "json"
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait_time = 5 * (2 ** attempt)
            print(f"TCIA API Error ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError(f"TCIA API Failed after {max_retries} retries")


# ────────────────────────────────────────────────────────────────────
# Download
# ────────────────────────────────────────────────────────────────────
def download_clinical_csv(out_dir):
    """Download C4KC-KiTS clinical CSV from TCIA."""
    out_path = out_dir / "clinical_data.csv"
    if out_path.exists():
        print(f"Clinical CSV already exists: {out_path}")
        return out_path

    print("Downloading C4KC-KiTS clinical CSV...")
    resp = requests.get(CLINICAL_CSV_URL, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Saved clinical CSV: {out_path} ({len(resp.content)} bytes)")
    return out_path


def get_all_series():
    """Get all series (CT + SEG) for C4KC-KiTS collection."""
    print("Querying TCIA for series...")
    series_list = tcia_get("getSeries", {"Collection": COLLECTION})

    ct_series = []
    seg_series = []

    for s in series_list:
        entry = {
            "SeriesInstanceUID": s["SeriesInstanceUID"],
            "PatientID": s["PatientID"],
            "Modality": s.get("Modality", ""),
            "SeriesDescription": s.get("SeriesDescription", ""),
            "ImageCount": s.get("ImageCount", 0),
        }
        if s.get("Modality") == "CT":
            ct_series.append(entry)
        elif s.get("Modality") == "SEG":
            seg_series.append(entry)

    ct_patients = set(s['PatientID'] for s in ct_series)
    seg_patients = set(s['PatientID'] for s in seg_series)

    print(f"Found {len(ct_series)} CT series ({len(ct_patients)} patients)")
    print(f"Found {len(seg_series)} SEG series ({len(seg_patients)} patients)")
    return ct_series, seg_series


def download_series(series_uid, out_dir):
    """Download a DICOM series as zip and extract."""
    url = f"{TCIA_BASE}/getImage"
    resp = requests.get(url, params={"SeriesInstanceUID": series_uid}, timeout=600, stream=True)
    resp.raise_for_status()

    content = BytesIO(resp.content)
    try:
        with zipfile.ZipFile(content) as zf:
            zf.extractall(out_dir)
        return True
    except zipfile.BadZipFile:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw_download.dcm").write_bytes(resp.content)
        return True


def download_all(ct_series, seg_series, dicom_base_dir):
    """Download CT and SEG DICOM series."""
    dicom_base_dir.mkdir(parents=True, exist_ok=True)

    # Group by patient
    patient_ct = {}
    for s in ct_series:
        pid = s["PatientID"]
        if pid not in patient_ct:
            patient_ct[pid] = []
        patient_ct[pid].append(s)

    patient_seg = {}
    for s in seg_series:
        pid = s["PatientID"]
        if pid not in patient_seg:
            patient_seg[pid] = []
        patient_seg[pid].append(s)

    all_patients = sorted(set(list(patient_ct.keys()) + list(patient_seg.keys())))
    success = 0
    skip = 0
    fail = 0

    for pid in tqdm(all_patients, desc="Downloading"):
        patient_dir = dicom_base_dir / pid
        ct_dir = patient_dir / "ct"
        seg_dir = patient_dir / "seg"

        # Skip already downloaded
        if ct_dir.exists() and len(list(ct_dir.rglob("*.dcm"))) > 5:
            skip += 1
            continue

        try:
            # Download CT(s) — pick arterial phase if multiple
            if pid in patient_ct:
                ct_dir.mkdir(parents=True, exist_ok=True)
                # Prefer arterial phase, else take first
                ct_candidates = patient_ct[pid]
                arterial = [c for c in ct_candidates if 'arter' in c.get('SeriesDescription', '').lower()]
                chosen = arterial[0] if arterial else ct_candidates[0]
                download_series(chosen["SeriesInstanceUID"], ct_dir)

            # Download SEG
            if pid in patient_seg:
                seg_dir.mkdir(parents=True, exist_ok=True)
                download_series(patient_seg[pid][0]["SeriesInstanceUID"], seg_dir)

            success += 1
        except Exception as e:
            print(f"\n  Failed {pid}: {e}")
            fail += 1

    print(f"\nDownload: {success} new, {skip} skipped, {fail} failed")


# ────────────────────────────────────────────────────────────────────
# DICOM → NIfTI
# ────────────────────────────────────────────────────────────────────
def convert_all_to_nifti(dicom_base_dir, nifti_dir):
    """Convert DICOM CTs and SEG to NIfTI."""
    nifti_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in dicom_base_dir.iterdir() if d.is_dir()])
    converted = 0
    skipped = 0

    for patient_dir in tqdm(patient_dirs, desc="DICOM → NIfTI"):
        patient_id = patient_dir.name
        out_dir = nifti_dir / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already converted
        if list(out_dir.glob("ct*.nii.gz")):
            skipped += 1
            continue

        # Convert CT
        ct_dir = patient_dir / "ct"
        if ct_dir.exists():
            cmd = [
                "dcm2niix",
                "-z", "y",
                "-f", "ct",
                "-o", str(out_dir),
                "-i", "y",
                str(ct_dir)
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=120)
            except Exception as e:
                print(f"\n  CT conversion error {patient_id}: {e}")

        # Convert SEG (DICOM SEG → NIfTI via dcmqi segimage2itkimage)
        seg_dir = patient_dir / "seg"
        if seg_dir.exists():
            seg_files = list(seg_dir.rglob("*.dcm"))
            if seg_files:
                seg_out_prefix = str(out_dir / "seg")
                try:
                    # Try segimage2itkimage (dcmqi)
                    subprocess.run([
                        "segimage2itkimage",
                        "--inputDICOM", str(seg_files[0]),
                        "--outputDirectory", str(out_dir),
                    ], capture_output=True, timeout=60)
                except FileNotFoundError:
                    # Fallback: try dcm2niix on seg directory
                    try:
                        subprocess.run([
                            "dcm2niix",
                            "-z", "y",
                            "-f", "seg",
                            "-o", str(out_dir),
                            str(seg_dir)
                        ], capture_output=True, timeout=60)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"\n  SEG conversion error {patient_id}: {e}")

        converted += 1

    print(f"\nConversion: {converted} new, {skipped} skipped")


# ────────────────────────────────────────────────────────────────────
# Resampling + Windowing + Slice extraction
# ────────────────────────────────────────────────────────────────────
def resample_isotropic(volume, original_spacing, target_spacing=TARGET_SPACING):
    """Resample to isotropic 1mm³ with linear interpolation."""
    if scipy_zoom is None:
        return volume
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    return scipy_zoom(volume, zoom_factors, order=1)


def resample_mask(mask, original_spacing, target_spacing=TARGET_SPACING):
    """Resample binary mask with nearest-neighbor."""
    if scipy_zoom is None:
        return mask
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    return scipy_zoom(mask, zoom_factors, order=0)


def get_voxel_spacing(nifti_img):
    """Extract voxel spacing from NIfTI header."""
    return np.array(nifti_img.header.get_zooms()[:3], dtype=float)


def apply_abdomen_window(image, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Apply abdominal soft tissue CT window: clip [-160, 240] HU.
    Standard for kidney/renal mass characterization (W=400, L=40).
    """
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min) * 255
    return image.astype(np.uint8)


def find_largest_tumor_slice(mask_data):
    """Find axial slice with largest tumor cross-section."""
    best_z = mask_data.shape[2] // 2
    best_area = 0
    for z in range(mask_data.shape[2]):
        area = np.sum(mask_data[:, :, z] > 0)
        if area > best_area:
            best_area = area
            best_z = z
    return best_z, best_area


def extract_all_slices(nifti_dir, images_dir, img_size=IMG_SIZE):
    """
    Steps for each patient:
      1. Load arterial-phase CT NIfTI
      2. Isotropic resampling to 1mm³
      3. Load kidney tumor segmentation → find largest tumor slice
      4. Apply abdominal soft tissue window [-160, 240] HU
      5. Resize to 224×224 → save as grayscale PNG
    """
    assert nib is not None, "pip install nibabel"

    images_dir.mkdir(parents=True, exist_ok=True)
    patient_dirs = sorted([d for d in nifti_dir.iterdir() if d.is_dir()])

    processed = 0
    skipped = 0
    stats = {"with_seg": 0, "without_seg": 0}

    for patient_dir in tqdm(patient_dirs, desc="Extracting slices"):
        patient_id = patient_dir.name
        out_path = images_dir / f"{patient_id}.png"

        if out_path.exists():
            processed += 1
            continue

        # Find CT NIfTI
        nii_files = list(patient_dir.glob("*.nii.gz"))
        ct_files = [f for f in nii_files if f.name.startswith("ct") or
                    ('seg' not in f.name.lower() and 'ct' in f.name.lower())]
        if not ct_files:
            ct_files = [f for f in nii_files if 'seg' not in f.name.lower()]
        if not ct_files:
            print(f"  No CT for {patient_id}")
            skipped += 1
            continue

        try:
            ct_nii = nib.load(str(ct_files[0]))
            ct_data = ct_nii.get_fdata().astype(np.float32)
            if ct_data.ndim > 3:
                ct_data = ct_data[:, :, :, 0]
            spacing = get_voxel_spacing(ct_nii)

            # Isotropic resampling
            ct_resampled = resample_isotropic(ct_data, spacing)

            # Find segmentation (fallback to middle slice)
            seg_files = [f for f in nii_files if 'seg' in f.name.lower()]
            if not seg_files:
                print(f"  Fallback {patient_id}: no tumor segmentation found")
                stats["without_seg"] += 1
                slice_idx = ct_resampled.shape[2] // 2
            else:
                mask_nii = nib.load(str(seg_files[0]))
                mask_data = mask_nii.get_fdata()
                if mask_data.ndim > 3:
                    mask_data = mask_data[:, :, :, 0]
                mask_resampled = resample_mask(mask_data, spacing)

                # Handle shape mismatch
                min_shape = np.minimum(ct_resampled.shape, mask_resampled.shape)
                mask_cropped = mask_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]
                slice_idx, tumor_area = find_largest_tumor_slice(mask_cropped)
                if tumor_area == 0:
                    print(f"  Fallback {patient_id}: segmentation mask is empty")
                    stats["without_seg"] += 1
                    slice_idx = ct_resampled.shape[2] // 2
                else:
                    stats["with_seg"] += 1

            slice_idx = min(slice_idx, ct_resampled.shape[2] - 1)

            # Extract axial slice + abdominal window
            axial_slice = ct_resampled[:, :, slice_idx]
            windowed = apply_abdomen_window(axial_slice)

            # Resize and save
            img = Image.fromarray(windowed, mode='L')
            img = img.resize((img_size, img_size), Image.LANCZOS)
            img.save(out_path)
            processed += 1

        except Exception as e:
            print(f"\n  Error for {patient_id}: {e}")
            skipped += 1

    print(f"\nExtraction: {processed} done, {skipped} skipped")
    print(f"Segmentation-guided: {stats['with_seg']}, middle-slice fallback: {stats['without_seg']}")


# ────────────────────────────────────────────────────────────────────
# Clinical CSV processing
# ────────────────────────────────────────────────────────────────────
def prepare_clinical_csv(raw_csv_path, out_csv_path):
    """
    Convert C4KC-KiTS clinical CSV → our standard format.
    Source columns: patient_id, vital_status, vital_days_after_surgery, malignant
    Output: patient_id, observed_time, event_indicator
    """
    df = pd.read_csv(raw_csv_path)

    # Filter to malignant tumors only
    df_mal = df[df['malignant'] == True].copy()
    print(f"Malignant cases: {len(df_mal)} / {len(df)} total")

    # Map vital_status → event indicator
    df_mal['event_indicator'] = df_mal['vital_status'].map({
        'dead': 1,
        'censored': 0,
        'alive': 0,       # treat alive as censored
    })

    out = pd.DataFrame({
        'patient_id': df_mal['patient_id'],
        'observed_time': df_mal['vital_days_after_surgery'],
        'event_indicator': df_mal['event_indicator'],
    })

    # Drop missing/invalid
    out = out.dropna(subset=['observed_time', 'event_indicator'])
    out['observed_time'] = pd.to_numeric(out['observed_time'], errors='coerce')
    out = out.dropna(subset=['observed_time'])
    out = out[out['observed_time'] > 0]

    out.to_csv(out_csv_path, index=False)
    n_events = int(out['event_indicator'].sum())
    print(f"Clinical CSV: {len(out)} malignant patients ({n_events} deaths, "
          f"{len(out)-n_events} censored) → {out_csv_path}")
    return out


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='C4KC-KiTS: Download + Preprocess')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Base output directory (e.g., $SCRATCH/Q-Surv/data/c4kc_kits)')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    parser.add_argument('--skip_download', action='store_true')
    parser.add_argument('--skip_convert', action='store_true')
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    raw_dir = base_dir / "raw"
    dicom_dir = raw_dir / "dicom"
    nifti_dir = raw_dir / "nifti"
    images_dir = base_dir / "images"

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("C4KC-KiTS Preprocessing Pipeline")
    print("  Preprocessing: 1mm³ isotropic → abdomen window [-160,240] HU")
    print("            → tumor seg guided slice → 224×224 PNG")
    print("=" * 60)

    # Step 1: Clinical CSV
    clinical_csv = download_clinical_csv(raw_dir)
    clinical_out = base_dir / "clinical_processed.csv"
    prepare_clinical_csv(str(clinical_csv), str(clinical_out))

    # Step 2: Download DICOMs
    if not args.skip_download:
        ct_series, seg_series = get_all_series()
        download_all(ct_series, seg_series, dicom_dir)
    else:
        print("Skipping TCIA download")

    # Step 3: DICOM → NIfTI
    if not args.skip_convert:
        convert_all_to_nifti(dicom_dir, nifti_dir)
    else:
        print("Skipping NIfTI conversion")

    # Step 4: Extract slices
    extract_all_slices(nifti_dir, images_dir, img_size=args.img_size)

    # Summary
    n_images = len(list(images_dir.glob("*.png")))
    n_clinical = len(pd.read_csv(str(clinical_out)))
    print(f"\n{'=' * 60}")
    print(f"C4KC-KiTS preprocessing complete!")
    print(f"  Images:   {n_images} PNGs in {images_dir}")
    print(f"  Clinical: {n_clinical} patients in {clinical_out}")
    print(f"  Preprocessing: 1mm³ isotropic, abdomen window [-160,240] HU,")
    print(f"            tumor-seg guided slice, {args.img_size}×{args.img_size} px")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
