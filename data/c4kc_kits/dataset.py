import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def get_c4kc_kits_dataset(seed=0, test_prop=0.2, train_transform=None, eval_transform=None, transform=None):
    """Return stratified-by-event (train_subset, test_subset) of the C4KC-KiTS dataset.

    `transform` is a legacy alias that fills both train_transform and eval_transform if neither is set.
    """
    if train_transform is None and transform is not None:
        train_transform = transform
    if eval_transform is None and transform is not None:
        eval_transform = transform

    csv_path = os.path.join('data/c4kc_kits', 'clinical_processed.csv')
    img_path_full = os.path.join('data/c4kc_kits', 'images')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run preprocessing first:\n"
            f"  python data/c4kc_kits/download_and_preprocess.py --out_dir data/c4kc_kits"
        )

    ds_train = C4KCKiTSSurvivalDataset(csv_path, img_path_full, transform=train_transform)
    ds_eval = C4KCKiTSSurvivalDataset(csv_path, img_path_full, transform=eval_transform)

    events = [s['event'] for s in ds_train.samples]
    indices = np.arange(len(ds_train))
    train_idx, test_idx = train_test_split(indices, test_size=test_prop, stratify=events, random_state=seed)
    return Subset(ds_train, train_idx), Subset(ds_eval, test_idx)


class C4KCKiTSSurvivalDataset(Dataset):
    def __init__(self, metadata_csv, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(metadata_csv)
        df = df[df['observed_time'] > 0]

        self.samples = []
        print(f"Loading C4KC-KiTS dataset from {metadata_csv} and {img_dir}...")
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            img_path = os.path.join(img_dir, f"{patient_id}.png")
            if not os.path.exists(img_path):
                continue
            try:
                time = float(row['observed_time'])
                event = int(row['event_indicator'])
            except ValueError:
                continue
            self.samples.append({'img_path': img_path, 'time': time, 'event': event,
                                 'patient_id': patient_id})
        print(f"Found {len(self.samples)} valid samples out of {len(df)} entries.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample['img_path']) as img:
            img = img.convert('L')
            if self.transform:
                img = self.transform(img)
            else:
                arr = np.array(img)
                img = arr[None, ...] if arr.ndim == 2 else arr.transpose(2, 0, 1)
        return img, torch.tensor(sample['time'], dtype=torch.float32), torch.tensor(sample['event'], dtype=torch.float32)
