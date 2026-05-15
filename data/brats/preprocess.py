import os
import glob
import SimpleITK as sitk
from PIL import Image
import numpy as np
import argparse
from multiprocessing import Pool
from functools import partial

def _pick_tumor_z(seg_arr):
    """Return z-index of axial slice with the largest tumor cross-section.
    seg_arr is (Z, Y, X) with 0 = background, >0 = tumor (any class).
    Returns None if mask is empty / not 3D."""
    if seg_arr.ndim != 3:
        return None
    per_slice_area = (seg_arr > 0).reshape(seg_arr.shape[0], -1).sum(axis=1)
    if per_slice_area.max() == 0:
        return None
    return int(np.argmax(per_slice_area))


def extract_largest_tumor_slice(nii_path, seg_path, out_path):
    """Extract the axial slice through the largest tumor cross-section
    using the BraTS segmentation mask. Falls back to centre slice if seg
    mask is missing or empty."""
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))  # (Z, Y, X)
        if arr.ndim == 2:
            slice_2d = arr
        elif arr.ndim == 3:
            z = None
            if seg_path is not None and os.path.exists(seg_path):
                try:
                    z = _pick_tumor_z(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))
                except Exception:
                    z = None
            if z is None:
                z = arr.shape[0] // 2
            slice_2d = arr[z, :, :]
        else:
            return False

        non_zero = slice_2d[slice_2d > 0]
        p99 = float(np.percentile(non_zero, 99)) if len(non_zero) else 1.0
        slice_2d = np.clip(slice_2d, 0, p99)
        slice_2d = (slice_2d / (p99 + 1e-8)) * 255.0
        slice_2d = slice_2d.astype(np.uint8)

        Image.fromarray(slice_2d).convert('L').save(out_path)
        return True
    except Exception as e:
        print(f"Error processing {nii_path}: {e}")
        return False


# Back-compat shim (was: center-slice only).
def extract_center_slice(nii_path, out_path, seg_path=None):
    return extract_largest_tumor_slice(nii_path, seg_path, out_path)


def process_file(f, args):
    basename = os.path.basename(f)
    # BraTS format: BraTS20_Training_001_flair.nii  → patient_id BraTS20_Training_001, modality flair
    parts = basename.replace('.nii.gz', '').replace('.nii', '').split('_')
    if len(parts) < 4:
        return 0
    modality = parts[-1]
    patient_id = "_".join(parts[:-1])

    out_name = f"{patient_id}_{modality}.png"
    out_path = os.path.join(args.out_dir, out_name)

    # Co-located seg mask: same patient, _seg.nii(.gz)
    nii_dir = os.path.dirname(f)
    seg_path = None
    for cand in (f"{patient_id}_seg.nii.gz", f"{patient_id}_seg.nii"):
        p = os.path.join(nii_dir, cand)
        if os.path.exists(p):
            seg_path = p
            break

    return 1 if extract_largest_tumor_slice(f, seg_path, out_path) else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/brats/MICCAI_BraTS2020_TrainingData', help='Directory with raw NIfTI files')
    parser.add_argument('--out_dir', type=str, default='data/brats/images', help='Directory for PNG slices')
    parser.add_argument('--modality', type=str, default='flair', help='Which MRI sequence to process (flair, t1, t1ce, t2)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of multiprocessing workers')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Process the specified modality
    nii_files = glob.glob(os.path.join(args.raw_dir, '**', f'*_{args.modality}.nii*'), recursive=True)
    
    if len(nii_files) == 0:
        print(f"No files matching *_{args.modality}.nii* found in {args.raw_dir}")
        exit(1)
        
    print(f"Found {len(nii_files)} files for modality '{args.modality}'. Starting pool with {args.num_workers} workers...")
    
    process_func = partial(process_file, args=args)
    
    with Pool(args.num_workers) as p:
        results = p.map(process_func, nii_files)
        
    success_count = sum(results)
    print(f"Done. Successfully extracted {success_count} {args.modality} PNGs to {args.out_dir}")
