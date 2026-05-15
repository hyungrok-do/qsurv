import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.base import SurvivalModel
from models.qsurv import get_quadrature_nodes_weights


class DeSurv(SurvivalModel):
    """Single-event DeSurv / DeCDF: network parameterizes g(x,t) = du/dt; F(t|x) = tanh(u(t|x))."""

    def __init__(self, network, n_nodes=15, optimizer_class=torch.optim.AdamW, scheduler_class=None,
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none'):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params,
                         scheduler_params, random_seed, batch_size, epochs, lr,
                         time_norm_mode)
        self.network = network
        self.n_nodes = n_nodes

        nodes, weights = get_quadrature_nodes_weights(n_nodes, method='legendre')
        self.register_buffer('nodes', nodes)
        self.register_buffer('weights', weights)

    def _positive_derivative(self, raw_output):
        # Network output is already softplus-positive; clamp for log-stability only.
        return raw_output.clamp(min=1e-8, max=30.0)

    def _format_times(self, x, t, per_sample):
        if t.ndim == 0:
            return t.view(1, 1).expand(x.shape[0], 1)

        if per_sample:
            if t.ndim == 1:
                return t.view(-1, 1)
            return t

        if t.ndim == 1:
            return t.view(1, -1).expand(x.shape[0], -1)
        if t.shape[0] == 1:
            return t.expand(x.shape[0], -1)
        return t

    def integrate_latent(self, x, t, per_sample=False):
        """Compute u(t|x) = int_0^t g(x,s) ds by Gauss-Legendre quadrature."""
        x = self._to_tensor(x)
        t = self._to_tensor(t)
        t_mat = self._format_times(x, t, per_sample=per_sample).clamp(min=0.0)

        batch_size = x.shape[0]
        n_times = t_mat.shape[1]
        nodes = self.nodes.to(t_mat.device)
        weights = self.weights.to(t_mat.device)

        half_width = 0.5 * t_mat
        midpoint = 0.5 * t_mat
        t_quad = half_width.unsqueeze(-1) * nodes.view(1, 1, -1) + midpoint.unsqueeze(-1)
        t_quad_flat = t_quad.reshape(-1, 1)

        if hasattr(self.network, 'precompute') and hasattr(self.network, 'forward_nodes_from_cache'):
            k_total = n_times * self.n_nodes
            h, y_static, v = self.network.precompute(x)
            g_vals = self.network.forward_nodes_from_cache(h, y_static, v, t_quad_flat, K=k_total)
        elif hasattr(self.network, 'forward_split'):
            g_vals = self.network.forward_split(x, t_quad_flat)
        else:
            scale = n_times * self.n_nodes
            x_exp = x.unsqueeze(1).expand(x.size(0), scale, *x.size()[1:]).reshape(-1, *x.size()[1:])
            g_vals = self.network(x_exp, t_quad_flat)

        g_vals = self._positive_derivative(g_vals.reshape(batch_size, n_times, self.n_nodes))
        u = half_width * torch.sum(weights.view(1, 1, -1) * g_vals, dim=2)
        return u

    def _eval_derivative(self, x, t, per_sample=False):
        x = self._to_tensor(x)
        t = self._to_tensor(t)
        t_mat = self._format_times(x, t, per_sample=per_sample).clamp(min=0.0)
        batch_size, n_times = t_mat.shape
        t_flat = t_mat.reshape(-1, 1)

        if hasattr(self.network, 'forward_split'):
            raw = self.network.forward_split(x, t_flat)
        else:
            x_exp = x.unsqueeze(1).expand(x.size(0), n_times, *x.size()[1:]).reshape(-1, *x.size()[1:])
            raw = self.network(x_exp, t_flat)

        return self._positive_derivative(raw.reshape(batch_size, n_times))

    def calculate_loss(self, x, t, e):
        e = e.view(-1).float()
        u = self.integrate_latent(x, t, per_sample=True).squeeze(-1)
        g_t = self._eval_derivative(x, t, per_sample=True).squeeze(-1)

        log_two = u.new_tensor(math.log(2.0))
        log_survival = log_two - torch.nn.functional.softplus(2.0 * u)
        log_sech2 = 2.0 * (log_two - u - torch.nn.functional.softplus(-2.0 * u))
        log_density = torch.log(g_t) + log_sech2

        log_likelihood = e * log_density + (1.0 - e) * log_survival
        return -log_likelihood.mean()

    def predict_survival_probability(self, X, t):
        self.eval()
        X = self._to_tensor(X)
        t = self._to_tensor(t)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        all_surv = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(next(self.parameters()).device)
                u = self.integrate_latent(batch_x, t, per_sample=False)
                surv = 1.0 - torch.tanh(u)
                all_surv.append(surv.clamp(0.0, 1.0).cpu())

        return torch.cat(all_surv, dim=0)

    def predict_hazard(self, X, t):
        """Implied hazard h(t|x) = g(x,t) * (1 + F(t|x)); DeSurv does not parameterize h directly."""
        self.eval()
        X = self._to_tensor(X)
        t = self._to_tensor(t)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        all_hazard = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(next(self.parameters()).device)
                u = self.integrate_latent(batch_x, t, per_sample=False)
                cdf = torch.tanh(u)
                g_t = self._eval_derivative(batch_x, t, per_sample=False)
                hazard = g_t * (1.0 + cdf)
                all_hazard.append(hazard.cpu())

        return torch.cat(all_hazard, dim=0)
