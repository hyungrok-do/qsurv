import torch
from torch.utils.data import Dataset
import numpy as np


class CoxCCDataset(Dataset):
    """For each event subject, sample n_controls from its time-descending risk set."""

    def __init__(self, x, t, e, n_controls=1, random_seed=None):
        def to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            return data

        x_in = to_numpy(x)
        t_in = to_numpy(t).flatten()
        e_in = to_numpy(e).flatten()
        self.n_controls = n_controls
        self.rng = np.random.RandomState(random_seed)

        sort_idx = np.argsort(t_in)[::-1].copy()
        self.x_sorted = x_in[sort_idx]
        self.t_sorted = t_in[sort_idx]
        self.e_sorted = e_in[sort_idx]
        self.sorted_event_indices = np.where(self.e_sorted > 0)[0]

    def __len__(self):
        return len(self.sorted_event_indices)

    def __getitem__(self, index):
        idx = self.sorted_event_indices[index]
        t_event = self.t_sorted[idx]

        # Risk set {j : t_j >= t_event}. In a time-descending array this is indices [0, risk_end).
        # searchsorted on -t_sorted with side='right' correctly includes tied event times.
        risk_end = np.searchsorted(-self.t_sorted, -t_event, side='right')

        if risk_end <= 1:
            control_indices = np.array([idx] * self.n_controls)
        else:
            cnt_idxs = []
            while len(cnt_idxs) < self.n_controls:
                needed = self.n_controls - len(cnt_idxs)
                cands = self.rng.choice(risk_end, needed, replace=(risk_end - 1 < needed))
                cnt_idxs.extend(cands[cands != idx])
            control_indices = np.array(cnt_idxs[:self.n_controls])

        return self.x_sorted[idx], self.x_sorted[control_indices]


class CoxTimeDataset(CoxCCDataset):
    """CoxCC dataset that additionally returns the case event time for time-dependent risk."""

    def __getitem__(self, index):
        x_case, x_controls = super().__getitem__(index)
        idx = self.sorted_event_indices[index]
        t_case = self.t_sorted[idx]
        return x_case, x_controls, np.array([t_case], dtype=np.float32)
