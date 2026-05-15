import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.base import SurvivalModel

class NnetSurv(SurvivalModel):
    def __init__(self, network, discretizer, optimizer_class=torch.optim.AdamW, scheduler_class=None, 
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none'):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr, time_norm_mode)
        self.network = network
        self.discretizer = discretizer
        if self.discretizer is None:
            raise ValueError("discretizer must be provided")

        self.register_buffer('cut_points_', torch.tensor([]))

    def calculate_loss(self, x, t, e):
        if self.cut_points_.numel() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        t = t.view(-1)
        t = t.clamp(min=self.cut_points_[0], max=self.cut_points_[-1] - 1e-6)
        
        indices = torch.bucketize(t, self.cut_points_, right=True) - 1
        n_bins = len(self.cut_points_) - 1
        indices = indices.clamp(0, n_bins - 1)
        
        logits = self.network(x)
        
        seq = torch.arange(n_bins, device=x.device).unsqueeze(0)
        idx_tensor = indices.unsqueeze(1)
        e_tensor = e.long().view(-1, 1)
        
        mask = seq <= idx_tensor
        mask = mask & ~((seq == idx_tensor) & (e_tensor == 0))
        
        target = torch.zeros_like(logits)
        target.scatter_(1, idx_tensor, e_tensor.float())
        
        loss_mat = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        masked_loss = loss_mat * mask.float()
        
        loss = masked_loss.sum() / (mask.sum() + 1e-7)
        
        return loss

    def fit(self, x, t, e, val_x=None, val_t=None, val_e=None):
        if isinstance(t, torch.Tensor):
            t_np = t.detach().cpu().numpy()
            e_np = e.detach().cpu().numpy()
        else:
            t_np = t
            e_np = e

        if self.discretizer is None:
             raise ValueError("discretizer cannot be None")

        edges = self.discretizer.bin_edges_
        if isinstance(edges, torch.Tensor):
             cut_points = edges.cpu().numpy()
        else:
             cut_points = np.array(edges)
        cut_points = np.unique(cut_points)
            
        max_t = t_np.max()
        if max_t > cut_points[-1]:
             cut_points = np.concatenate([cut_points, [max_t]])
             
        span = cut_points[-1] - cut_points[0]
        padding = span * 0.05 if span > 0 else 1.0
        cut_points = np.concatenate([cut_points, [cut_points[-1] + padding]])

        device = next(self.parameters()).device
        self.cut_points_ = torch.tensor(cut_points, dtype=torch.float32).to(device)
        n_bins = len(self.cut_points_) - 1

        model_inner = self.network
        if hasattr(model_inner, 'net'):
            model_inner = model_inner.net

        if isinstance(model_inner, nn.Sequential):
            last_layer = model_inner[-1]
            if isinstance(last_layer, nn.Linear) and last_layer.out_features != n_bins:
                model_inner[-1] = nn.Linear(last_layer.in_features, n_bins).to(device)
        elif hasattr(model_inner, 'head') and isinstance(model_inner.head, nn.Linear):
            if model_inner.head.out_features != n_bins:
                model_inner.head = nn.Linear(model_inner.head.in_features, n_bins).to(device)

        return super().fit(x, t, e, val_x, val_t, val_e)

    def predict_survival_probability(self, x, t):
        self.eval()
        
        if isinstance(x, DataLoader):
            loader = x
        elif isinstance(x, torch.utils.data.Dataset):
            loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        else:
            x = self._to_tensor(x)
            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            
        t = torch.tensor(t) if not isinstance(t, torch.Tensor) else t
        t = t.to(next(self.parameters()).device)
        
        with torch.no_grad():
            all_probs = []

            if t.ndim == 0: t = t.unsqueeze(0)

            # Step-function survival: S(t) = S(c[j-1]) for t in [c[j-1], c[j]).
            indices = torch.bucketize(t, self.cut_points_, right=True) - 1
            indices = indices.clamp(0, len(self.cut_points_) - 2)

            for batch_x in loader:
                batch_x = batch_x[0].to(next(self.parameters()).device)

                logits = self.network(batch_x)
                hazards = torch.sigmoid(logits)

                surv_at_bins = torch.cumprod(1 - hazards, dim=1)

                ones = torch.ones(batch_x.shape[0], 1, device=batch_x.device)
                surv_all = torch.cat([ones, surv_at_bins], dim=1)

                surv_probs = surv_all[:, indices]
                all_probs.append(surv_probs.cpu())

            return torch.cat(all_probs, dim=0)

    def predict_hazard(self, x, t):
        """Discrete per-bin hazard probability for the bin containing t."""
        self.eval()
        x = self._to_tensor(x)
        t = self._to_tensor(t)

        with torch.no_grad():
            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_hazard = []

            if t.ndim == 0: t = t.unsqueeze(0)

            indices = torch.bucketize(t, self.cut_points_, right=True) - 1
            indices = indices.clamp(0, len(self.cut_points_) - 2)

            for batch_x in loader:
                batch_x = batch_x[0].to(x.device)

                logits = self.network(batch_x)
                hazards = torch.sigmoid(logits)

                sel_hazards = hazards[:, indices]
                all_hazard.append(sel_hazards.cpu())

            return torch.cat(all_hazard, dim=0)
