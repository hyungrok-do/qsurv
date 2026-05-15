import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=32, mu=0.0, sigma=1.0, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))
        self.eps = eps
        self.register_buffer("freqs", torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)))

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = (t - self.mu) / (self.sigma + self.eps)
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeNormalizer(nn.Module):
    """Normalizes time values to the network's input scale.

    With min_max normalization (mu=0, sigma=t_max), t in [0, t_max] maps to v in [0, 1]
    so that integrating from t=0 corresponds to v=0 with no negative-v extrapolation.
    """

    def __init__(self, mu=0.0, sigma=1.0, time_norm_mode='min_max', eps=1e-8):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))
        self.time_norm_mode = time_norm_mode
        self.eps = eps

    def forward(self, t):
        if self.time_norm_mode == 'log_std':
            # log1p requires t >= 0 (always true for survival times).
            return (torch.log1p(t) - self.mu) / (self.sigma + self.eps)
        if self.time_norm_mode in ['percentile', 'identity_std', 'quantile', 'min_max']:
            return (t - self.mu) / (self.sigma + self.eps)
        return t
