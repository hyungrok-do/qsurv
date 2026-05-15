import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

def get_brats_dataset(seed=0, test_prop=0.2, train_transform=None, eval_transform=None, transform=None, modality="flair"):
    if train_transform is None and transform is not None:
        train_transform = transform
    if eval_transform is None and transform is not None:
        eval_transform = transform

    data_dir = 'data/brats'
    csv_file = 'survival_info.csv'
    img_dir = 'images'

    csv_path = os.path.join(data_dir, csv_file)
    img_path_full = os.path.join(data_dir, img_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Ensure the BraTS survival metadata is present.")

    ds_train = BRATSSurvivalDataset(csv_path, img_path_full, transform=train_transform, modality=modality)
    ds_eval = BRATSSurvivalDataset(csv_path, img_path_full, transform=eval_transform, modality=modality)

    # BraTS survival info might have 'ALIVE' vs 'DEAD', event = 1 if DEAD
    events = [s['event'] for s in ds_train.samples]
    indices = np.arange(len(ds_train))

    try:
        train_idx, test_idx = train_test_split(indices, test_size=test_prop, stratify=events, random_state=seed)
    except ValueError:
        print("Warning: Could not stratify by event (too few censored samples). Falling back to random split.")
        train_idx, test_idx = train_test_split(indices, test_size=test_prop, random_state=seed)

    train_final = Subset(ds_train, train_idx)
    test_final = Subset(ds_eval, test_idx)

    return train_final, test_final

class BRATSSurvivalDataset(Dataset):
    def __init__(self, metadata_csv, img_dir, transform=None, modality="flair"):
        self.img_dir = img_dir
        self.transform = transform
        self.modality = modality

        df = pd.read_csv(metadata_csv)
        self.samples = []
        print(f"Loading BraTS dataset from {metadata_csv}...")

        for _, row in df.iterrows():
            patient_id = str(row['Brats20ID']).strip()
            img_path = os.path.join(img_dir, f"{patient_id}_{self.modality}.png")
            if not os.path.exists(img_path):
                continue
            try:
                val = str(row['Survival_days']).strip().upper()
                if 'ALIVE' in val:
                    # ALIVE rows store the last-known follow-up days in the same cell.
                    m = re.search(r'\d+', val)
                    if not m:
                        continue
                    time, event = float(m.group()), 0
                else:
                    time, event = float(val), 1
            except ValueError:
                continue
            if time <= 0:
                continue
            self.samples.append({'img_path': img_path, 'time': time, 'event': event,
                                 'patient_id': patient_id})
        print(f"Found {len(self.samples)} valid samples out of {len(df)} entries.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        
        # FLAIR is intrinsically single-channel; load as L. The backbone replaces
        # conv1 with a 1-channel layer whose weights are the mean across the 3
        # ImageNet RGB channels — better than feeding [I, I, I] into RGB filters.
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)

        time = torch.tensor(sample['time'], dtype=torch.float32)
        event = torch.tensor(sample['event'], dtype=torch.float32)
        
        return image, time, event
