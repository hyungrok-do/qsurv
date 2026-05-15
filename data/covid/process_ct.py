# preprocess_dicom_to_png.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

# Paths
IMG_ROOT = "./COVID-19-NY-SBU"              # Root directory of original DICOM data
OUT_ROOT = "./images"      # Output directory for PNG files
CSV_PATH = "clinical_processed.csv"         # Clinical CSV file path

os.makedirs(OUT_ROOT, exist_ok=True)

VALID_STUDY_TYPES = [
    "CHEST AP PORT",
    "CHEST AP VIEWONLY",
    "CHEST AP PORTABLE",
    "CHEST AP VIEW ONLY",
]


def find_best_study_dir(patient_id: str, visit_dt: pd.Timestamp):

    patient_dir = os.path.join(IMG_ROOT, str(patient_id))
    if not os.path.isdir(patient_dir):
        return None

    vdate = visit_dt.date()
    candidates = []

    for name in os.listdir(patient_dir):
        # Use CHEST AP-related folders only
        if not any(t in name for t in VALID_STUDY_TYPES):
            continue

        parts = name.split("-")
        if len(parts) < 3:
            continue

        date_str = "-".join(parts[:3])  # "MM-DD-YYYY"
        try:
            img_date = datetime.strptime(date_str, "%m-%d-%Y").date()
        except ValueError:
            continue

        diff_days = (img_date - vdate).days
        candidates.append((abs(diff_days), diff_days, name))

    if not candidates:
        return None

    # Choose the folder with the smallest absolute date difference to the visit date
    candidates.sort(key=lambda x: x[0])  # (abs_diff, diff, name)
    best_name = candidates[0][2]
    return os.path.join(patient_dir, best_name)


def find_first_sequence_dir(study_dir: str):

    if not os.path.isdir(study_dir):
        return None

    seqs = []
    for name in os.listdir(study_dir):
        full = os.path.join(study_dir, name)
        if not os.path.isdir(full):
            continue

        # Parse the numeric prefix before the first '-'
        head = name.split("-", 1)[0]
        try:
            num = float(head)
        except ValueError:
            continue
        seqs.append((num, full))

    if not seqs:
        return None

    seqs.sort(key=lambda x: x[0])
    return seqs[0][1]  # The sequence folder with the smallest numeric prefix


def load_main_dicom(seq_dir: str):

    if seq_dir is None or not os.path.isdir(seq_dir):
        return None

    dicom_path = os.path.join(seq_dir, "1-1.dcm")
    if not os.path.isfile(dicom_path):
        # Fallback: use the first DICOM file if '1-1.dcm' does not exist
        dcm_files = [f for f in os.listdir(seq_dir) if f.lower().endswith(".dcm")]
        if not dcm_files:
            return None
        dicom_path = os.path.join(seq_dir, sorted(dcm_files)[0])

    try:
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def resize_and_pad_256(arr: np.ndarray) -> np.ndarray:
    """
    Resize the image so that the longer side becomes 256, then zero-pad the shorter side
    to produce a (256, 256) array (center-aligned).

    Input:
        arr: (H, W) float32
    Output:
        canvas: (256, 256) float32
    """
    H, W = arr.shape
    long_side = max(H, W)
    scale = 256.0 / long_side
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))

    # Resize (bilinear)
    img = Image.fromarray(arr)
    img_resized = img.resize((new_W, new_H), Image.BILINEAR)
    arr_resized = np.array(img_resized, dtype=np.float32)

    # Zero-pad to 256x256 (center alignment)
    canvas = np.zeros((256, 256), dtype=np.float32)
    top = (256 - new_H) // 2
    left = (256 - new_W) // 2
    canvas[top:top + new_H, left:left + new_W] = arr_resized

    return canvas


def save_array_as_png_gray(arr: np.ndarray, out_path: str):

    a = arr.astype(np.float32)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))

    if mx <= mn:
        norm = np.zeros_like(a, dtype=np.uint8)
    else:
        norm = (a - mn) / (mx - mn)
        norm = (norm * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(norm, mode="L")  # 8-bit grayscale
    img.save(out_path)

# =========================
# Main
# =========================
df = pd.read_csv(CSV_PATH)
df["visit_start_datetime"] = pd.to_datetime(df["visit_start_datetime"])

num_ok = 0
num_fail = 0

for _, row in df.iterrows():
    pid = str(row["to_patient_id"])
    visit_dt = row["visit_start_datetime"]

    png_path = os.path.join(OUT_ROOT, f"{pid}.png")

    # 1) Find the best-matching CHEST AP study folder
    study_dir = find_best_study_dir(pid, visit_dt)
    if study_dir is None:
        print(f"[WARN] No valid study for patient {pid}")
        num_fail += 1
        continue

    # 2) Find the earliest sequence folder in that study
    seq_dir = find_first_sequence_dir(study_dir)
    if seq_dir is None:
        print(f"[WARN] No valid sequence in {study_dir} for patient {pid}")
        num_fail += 1
        continue

    # 3) Load the main DICOM image
    arr = load_main_dicom(seq_dir)
    if arr is None:
        print(f"[WARN] Cannot load DICOM for patient {pid}")
        num_fail += 1
        continue

    # 4) Resize + pad to 256x256
    processed = resize_and_pad_256(arr)

    # 5) Save PNG
    save_array_as_png_gray(processed, png_path)
    print(f"[OK] Saved {png_path}")
    num_ok += 1

print(f"Done. ok={num_ok}, fail={num_fail}")
