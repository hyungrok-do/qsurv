import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base import SurvivalModel
from torch.utils.data import DataLoader, TensorDataset


def get_quadrature_nodes_weights(n_nodes, method='legendre', device='cpu'):
    """Gauss quadrature nodes and weights on [-1, 1]."""
    if method == 'legendre':
        nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
    return torch.from_numpy(nodes).float().to(device), torch.from_numpy(weights).float().to(device)


class QSurv(SurvivalModel):
    def __init__(self, network, n_nodes=15, optimizer_class=torch.optim.AdamW, scheduler_class=None,
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none'):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr, time_norm_mode)
        self.network = network
        self.n_nodes = n_nodes

        nodes, weights = get_quadrature_nodes_weights(n_nodes, method='legendre')
        self.register_buffer('nodes', nodes)
        self.register_buffer('weights', weights)

    def _apply_hazard_activation(self, raw_output):
        return raw_output.clamp(min=1e-8, max=30)

    def forward(self, x, t):
        """Returns (hazard, cum_hazard); the network outputs positive hazard via softplus."""
        if t.ndim == 1:
            t = t.view(-1, 1)
        raw_output = self.network(x, t)
        hazard = self._apply_hazard_activation(raw_output)
        cum_hazard = self.integrate_cumulative_hazard(x, t)
        return hazard, cum_hazard

    def integrate_cumulative_hazard(self, x, t):
        """Integrate hazard h(s) from 0 to t via Gauss-Legendre quadrature.

        Quadrature is set up in the network's normalized time space and the nodes are
        mapped back to raw time before being fed to the network. With normalization
        v = (raw - mu) / sigma, the bounds are v0 = -mu/sigma (raw=0) and vT = (t-mu)/sigma.
        """
        batch_size = x.shape[0]
        if t.ndim == 1:
            t = t.unsqueeze(1)
        n_times = t.shape[1]
        nodes = self.nodes
        weights = self.weights
        K = self.n_nodes

        mu = getattr(self.network, 'mu', 0.0)
        sigma = getattr(self.network, 'sigma', 1.0)
        if not isinstance(mu, torch.Tensor):
            mu = t.new_tensor(mu)
        if not isinstance(sigma, torch.Tensor):
            sigma = t.new_tensor(sigma)
        sigma_safe = sigma + 1e-8

        v0 = (0 - mu) / sigma_safe
        vT = (t - mu) / sigma_safe                                              # (B, T)
        vT = vT.clamp(min=v0)

        half_sum = (vT + v0) * 0.5
        half_diff = (vT - v0) * 0.5

        v_quad = half_diff.unsqueeze(-1) * nodes.view(1, 1, -1) + half_sum.unsqueeze(-1)
        v_quad_flat = v_quad.reshape(-1, 1)
        t_quad_flat = (v_quad_flat * sigma + mu).clamp(min=0)

        if hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
            K_total = n_times * K
            h, y_static, v = self.network.precompute(x)
            hazard_vals = self.network.forward_nodes_from_cache(h, y_static, v, t_quad_flat, K=K_total)
        elif hasattr(self.network, 'forward_split'):
            hazard_vals = self.network.forward_split(x, t_quad_flat)
        else:
            scale = n_times * K
            x_exp = x.unsqueeze(1).expand(x.size(0), scale, *x.size()[1:]).reshape(-1, *x.size()[1:])
            hazard_vals = self.network(x_exp, t_quad_flat)

        hazard_vals = self._apply_hazard_activation(hazard_vals.reshape(batch_size, n_times, K))
        weighted_sum = torch.sum(weights.view(1, 1, -1) * hazard_vals, dim=2)   # (B, T)

        # half_diff * sigma rescales the [-1,1] quadrature back to raw time units.
        return half_diff * sigma * weighted_sum

    def calculate_loss(self, x, t, e):
        """NLL: -mean(e * log h(t) - H(t)). Hazard is clamped in forward(), so log is safe."""
        hazard, cum_hazard = self.forward(x, t)
        hazard = hazard.squeeze(-1)
        cum_hazard = cum_hazard.squeeze(-1)
        e = e.view(-1)
        return -torch.mean(e * torch.log(hazard) - cum_hazard)

    def predict_hazard(self, X, t):
        self.eval()
        X = self._to_tensor(X)
        t = self._to_tensor(t)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        n_times = t.shape[0]
        device = next(self.network.parameters()).device
        t = t.to(device)

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        all_haz = []
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(device)
                b_size = batch_X.shape[0]
                if hasattr(self.network, 'forward_split'):
                    t_rep = t.unsqueeze(0).expand(b_size, -1).reshape(-1, 1)
                    raw_output = self.network.forward_split(batch_X, t_rep)
                else:
                    batch_X_expanded = batch_X.unsqueeze(1).expand(batch_X.size(0), n_times, *batch_X.size()[1:]).reshape(-1, *batch_X.size()[1:])
                    batch_t_expanded = t.unsqueeze(0).expand(b_size, n_times).reshape(-1, 1)
                    raw_output = self.network(batch_X_expanded, batch_t_expanded)
                hazard = self._apply_hazard_activation(raw_output.reshape(b_size, n_times))
                all_haz.append(hazard.cpu())
        return torch.cat(all_haz, dim=0)

    def predict_survival_probability(self, X, t):
        self.eval()
        X = self._to_tensor(X)
        t = self._to_tensor(t)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        n_times = t.shape[0]
        device = next(self.network.parameters()).device
        t = t.to(device)

        mu = getattr(self.network, 'mu', 0.0)
        sigma = getattr(self.network, 'sigma', 1.0)
        if not isinstance(mu, torch.Tensor):
            mu = t.new_tensor(mu)
        if not isinstance(sigma, torch.Tensor):
            sigma = t.new_tensor(sigma)

        # Sorted grid lets us reuse the cumulative integral instead of integrating [0, t_k]
        # independently for each k.
        is_sorted_grid = n_times > 1 and torch.all(t[1:] >= t[:-1])

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        all_surv = []
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(device)
                b_size = batch_X.shape[0]

                if is_sorted_grid:
                    grid_with_zero = torch.cat([torch.zeros(1, device=t.device), t])
                    diff_t = grid_with_zero[1:] - grid_with_zero[:-1]            # (T,)
                    mid_t = (grid_with_zero[1:] + grid_with_zero[:-1]) / 2

                    # Quadrature nodes in raw time; the network's TimeNormalizer handles normalization.
                    t_nodes = (diff_t.unsqueeze(1) / 2) * self.nodes.view(1, -1) + mid_t.unsqueeze(1)
                    t_nodes_flat = t_nodes.reshape(-1, 1).clamp(min=0)
                    t_quad_batch = t_nodes_flat.unsqueeze(0).expand(b_size, -1, -1).reshape(-1, 1)

                    if hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
                        K_total = n_times * self.n_nodes
                        h_pre, y_static, v_pre = self.network.precompute(batch_X)
                        h_vals = self.network.forward_nodes_from_cache(h_pre, y_static, v_pre, t_quad_batch, K=K_total)
                    elif hasattr(self.network, 'forward_split'):
                        h_vals = self.network.forward_split(batch_X, t_quad_batch)
                    else:
                        scale = n_times * self.n_nodes
                        batch_X_exp = batch_X.unsqueeze(1).expand(batch_X.size(0), scale, *batch_X.size()[1:]).reshape(-1, *batch_X.size()[1:])
                        h_vals = self.network(batch_X_exp, t_quad_batch)

                    h_vals = self._apply_hazard_activation(h_vals.view(b_size, n_times, self.n_nodes))
                    delta_H = (diff_t.view(1, -1) / 2) * torch.sum(self.weights.view(1, 1, -1) * h_vals, dim=2)
                    cum_hazard = torch.cumsum(delta_H, dim=1)
                else:
                    batch_t_expanded = t.unsqueeze(0).expand(b_size, -1)
                    cum_hazard = self.integrate_cumulative_hazard(batch_X, batch_t_expanded)

                surv = torch.exp(-cum_hazard.clamp(max=30))
                all_surv.append(surv.reshape(b_size, n_times).cpu())
        return torch.cat(all_surv, dim=0)

    def predict_cumulative_hazard(self, X, t):
        self.eval()
        X = self._to_tensor(X)
        t = self._to_tensor(t)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        n_times = t.shape[0]
        device = next(self.network.parameters()).device
        t = t.to(device)

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        all_cum_haz = []
        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(device)
                b_size = batch_X.shape[0]
                batch_t_expanded = t.unsqueeze(0).expand(b_size, -1)
                cum_hazard = self.integrate_cumulative_hazard(batch_X, batch_t_expanded)
                all_cum_haz.append(cum_hazard.reshape(b_size, n_times).cpu())
        return torch.cat(all_cum_haz, dim=0)
