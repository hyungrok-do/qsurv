import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.base import SurvivalModel, eps

class DeepHitLoss(torch.nn.Module):
    """DeepHit loss = (1 - alpha) * NLL + alpha * ranking; supports sample weights."""

    def __init__(self, alpha=0.5, sigma=1.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.alpha = alpha
        self.sigma = sigma
        self.eps = 1e-6

    def _nll_loss(self, event_prob, cumul_prob, t_one_hot, e, weights=None):
        event_prob_at_t = (event_prob * t_one_hot).sum(dim=1)
        cumul_prob_at_t = (cumul_prob * t_one_hot).sum(dim=1)
        log_likelihood = (
            e * torch.log(torch.clamp(event_prob_at_t, min=self.eps))
            + (1 - e) * torch.log(torch.clamp(1.0 - cumul_prob_at_t, min=self.eps))
        )
        if weights is not None:
            return -(log_likelihood * weights).sum() / torch.clamp(weights.sum(), min=self.eps)
        return -log_likelihood.mean()

    def _ranking_loss(self, cumul_prob, t_one_hot, t_raw, e, weights=None):
        # C_matrix[j, i] = F(t_i | x_j); a valid rank pair has row=j (later subject), col=i (earlier event).
        pair_mask = (e.view(1, -1) == 1) & (t_raw.view(1, -1) < t_raw.view(-1, 1))
        num_pairs = pair_mask.sum()
        if num_pairs.item() < 1:
            return torch.tensor(0.0, device=cumul_prob.device)

        C_matrix = cumul_prob @ t_one_hot.float().T
        F_i_at_ti = torch.diag(C_matrix).unsqueeze(0)
        diff_matrix = C_matrix - F_i_at_ti
        ranking_terms = torch.exp(diff_matrix / self.sigma) * pair_mask.float()

        if weights is not None:
            # Weights belong to the earlier event subject i, indexed along columns.
            weight_matrix = weights.view(1, -1)
            denominator = (pair_mask.float() * weight_matrix).sum()
            return (ranking_terms * weight_matrix).sum() / torch.clamp(denominator, min=self.eps)
        return ranking_terms.sum() / torch.clamp(num_pairs.float(), min=self.eps)


    def forward(self, event_prob, time_onehot, time_raw, event, weights=None):
        event = event.view(-1).float()
        time_raw = time_raw.view(-1)
        if weights is not None:
            weights = weights.view(-1).float()

        cumul_prob = torch.cumsum(event_prob, dim=1)
        
        nll = self._nll_loss(event_prob, cumul_prob, time_onehot, event, weights)
        ranking = torch.tensor(0.0, device=event_prob.device)
        if self.alpha > 0:
            ranking = self._ranking_loss(cumul_prob, time_onehot, time_raw, event, weights)

        return (1.0 - self.alpha) * nll + self.alpha * ranking


class DeepHit(SurvivalModel):
    def __init__(self, network, discretizer, alpha=0.5, sigma=2, optimizer_class=torch.optim.AdamW, scheduler_class=None, 
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr, 'none')
        self.network = network
        self.discretizer = discretizer
        if self.discretizer is None:
            raise ValueError("discretizer must be provided")
        self.alpha = alpha
        self.sigma = sigma

        self.register_buffer('cut_points_', torch.tensor([]))
        
        self.loss_fn = DeepHitLoss(alpha=alpha, sigma=sigma)

    def calculate_loss(self, x, t, e):
        t_raw = t.view(-1)
        if self.cut_points_.numel() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        t_clamped = t_raw.clamp(min=self.cut_points_[0], max=self.cut_points_[-1] - 1e-6)
        
        indices = torch.bucketize(t_clamped, self.cut_points_, right=True) - 1
        n_bins = len(self.cut_points_) - 1
        indices = indices.clamp(0, n_bins - 1)
        
        time_onehot = F.one_hot(indices, num_classes=n_bins).float()
        
        # Networks output logits; softmax converts them into a valid PMF.
        pmf = F.softmax(self.network(x), dim=1)
        return self.loss_fn(pmf, time_onehot, t_raw, e)
        
    def fit(self, x, t, e, val_x=None, val_t=None, val_e=None):
        if isinstance(t, torch.Tensor):
            t_np = t.detach().cpu().numpy()
            e_np = e.detach().cpu().numpy()
        else:
            t_np = t
            e_np = e
            
        if self.discretizer is None:
            raise ValueError("discretizer must be provided")

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

        t_tensor = torch.tensor(t) if not isinstance(t, torch.Tensor) else t
        t_tensor = t_tensor.float().to(next(self.parameters()).device)
        
        with torch.no_grad():
            all_probs = []
            
            if t_tensor.ndim == 0: t_tensor = t_tensor.unsqueeze(0)
            
            indices = torch.bucketize(t_tensor, self.cut_points_, right=True) - 1
            indices = indices.clamp(0, len(self.cut_points_) - 2)
            
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                    
                batch_x = batch_x.to(next(self.parameters()).device)
                
                pmf = self.network(batch_x)
                if not torch.is_tensor(pmf):
                    pmf = torch.tensor(pmf, device=batch_x.device)
                # Networks output logits; softmax converts them into a valid PMF.
                pmf = F.softmax(pmf, dim=1)
                cdf = torch.cumsum(pmf, dim=1)
                surv = 1.0 - cdf
                
                ones = torch.ones(batch_x.shape[0], 1, device=batch_x.device)
                surv_all = torch.cat([ones, surv], dim=1)
                
                probs = surv_all[torch.arange(batch_x.shape[0]).unsqueeze(1), indices]
                all_probs.append(probs.cpu())
                
            return torch.cat(all_probs, dim=0)
