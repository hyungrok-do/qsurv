import torch
import numpy as np

class LabTransDiscreteTime:
    def __init__(self, num_durations, scheme='quantile', extra_bins=1):
        if scheme not in ['quantile', 'uniform']:
            raise ValueError("scheme must be 'quantile' or 'uniform'")
        self.num_durations = num_durations
        self.scheme = scheme
        self.extra_bins = extra_bins
        self.bin_edges_ = None

    def fit(self, durations):
        if isinstance(durations, np.ndarray):
            durations = torch.as_tensor(durations, dtype=torch.float32)
        durations = durations.flatten()

        def compute_edges(d):
            if self.scheme == 'quantile':
                quantiles = torch.linspace(0, 1, self.num_durations + 1, device=d.device, dtype=d.dtype)
                return torch.quantile(d, quantiles)
            elif self.scheme == 'uniform':
                min_val = torch.min(d)
                max_val = torch.max(d)
                return torch.linspace(min_val, max_val + 1e-8, self.num_durations + 1, device=d.device)

        self.bin_edges_ = compute_edges(durations)
        unique_edges = torch.unique(self.bin_edges_)

        # Local-generator jitter (not global torch RNG) keeps bin edges deterministic on
        # datasets with many tied durations.
        if unique_edges.numel() - 1 < self.num_durations:
            eps = 1e-4 * torch.mean(durations).item() + 1e-6
            gen = torch.Generator(device=durations.device).manual_seed(0)
            jitter = torch.rand(durations.shape, generator=gen,
                                device=durations.device, dtype=durations.dtype) * eps
            self.bin_edges_ = compute_edges(durations + jitter)
            unique_edges = torch.unique(self.bin_edges_)

        # Uniform fallback if quantiles still collapse: callers rely on num_durations bins.
        if unique_edges.numel() - 1 < self.num_durations:
            min_val = torch.min(durations)
            max_val = torch.max(durations)
            self.bin_edges_ = torch.linspace(min_val, max_val + 1e-5, self.num_durations + 1, device=durations.device)
            unique_edges = self.bin_edges_

        self.bin_edges_ = unique_edges
        return self

    def transform(self, durations):
        # 1-based index, clamped to [1, num_durations]; bin num_durations+1 stays empty (reserved for > t_max).
        if isinstance(durations, torch.Tensor):
            d = durations.flatten()
            if isinstance(self.bin_edges_, torch.Tensor):
                bin_rights = self.bin_edges_[1:].to(device=d.device, dtype=d.dtype)
            else:
                bin_rights = torch.as_tensor(self.bin_edges_[1:], device=d.device, dtype=d.dtype)
            duration_idx = (d.unsqueeze(1) >= bin_rights).sum(dim=1) + 1
            return duration_idx.clamp(1, self.num_durations)

        if isinstance(durations, np.ndarray):
            d = durations.reshape(-1)
            if isinstance(self.bin_edges_, np.ndarray):
                bin_rights = self.bin_edges_[1:]
            else:
                bin_rights = np.asarray(self.bin_edges_[1:], dtype=d.dtype)
            duration_idx = (d[:, None] >= bin_rights).sum(axis=1) + 1
            return np.clip(duration_idx, 1, self.num_durations).astype(np.int64)

        raise TypeError("durations must be a torch.Tensor or numpy.ndarray")

    def transform_one_hot(self, durations):
        duration_idx = self.transform(durations)
        return self.one_hot(duration_idx)

    def one_hot(self, duration_idx):
        n_samples = duration_idx.shape[0]
        max_bins = self.num_durations + self.extra_bins
        out = torch.zeros((n_samples, max_bins), dtype=torch.float32, device=duration_idx.device)

        for i in range(n_samples):
            idx = duration_idx[i]
            if 1 <= idx <= max_bins:
                out[i, idx - 1] = 1

        return out

    def cumulative_one_hot(self, duration_idx):
        n_samples = duration_idx.shape[0]
        max_bins = self.num_durations + self.extra_bins
        out = torch.zeros((n_samples, max_bins), dtype=torch.float32, device=duration_idx.device)

        for i in range(n_samples):
            idx = duration_idx[i]
            if idx > 0:
                out[i, :idx] = 1

        return out

    def inverse_transform(self, duration_idx):
        duration_idx = torch.clamp(duration_idx, 1, self.num_durations)
        return self.bin_edges_[duration_idx]

    def fit_transform(self, durations):
        return self.fit(durations).transform(durations)

    def get_bin_edges(self):
        return self.bin_edges_
