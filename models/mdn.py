import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.base import SurvivalModel
from torch.utils.data import DataLoader, TensorDataset

def safe_inverse_softplus(x):
    """Inverse softplus log(exp(x) - 1); falls back to identity for x > 20 where exp overflows."""
    return torch.where(
        x > 20.0,
        x,
        torch.log(torch.exp(x.clamp(max=20.0)) - 1 + 1e-6),
    )


def inverse_softplus_grad(x):
    """Gradient of inverse softplus; clamped argument to avoid exp overflow."""
    return torch.where(
        x > 20.0,
        torch.ones_like(x),
        torch.exp(x.clamp(max=20.0)) / (torch.exp(x.clamp(max=20.0)) - 1 + 1e-6),
    )


def logsumexp_weighted(a, dim, b):
    """Weighted logsumexp log(sum(b * exp(a))); supports negative weights b."""
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-6)
    out += a_max
    return out


class MDN(SurvivalModel):
    """Survival Mixture Density Network.

    Reference: XintianHan/Survival-MDN. Models T = softplus(T*) with T* ~ Gaussian
    mixture; loss applies a Jacobian correction for the inverse-softplus transform.
    """

    def __init__(self, network, optimizer_class=torch.optim.AdamW, scheduler_class=None,
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none'):
        super().__init__(network, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr, time_norm_mode)

        if not hasattr(self.network, 'mu'):
            self.network.register_buffer('mu', torch.tensor(0.0))
        if not hasattr(self.network, 'sigma'):
            self.network.register_buffer('sigma', torch.tensor(1.0))

    def _sanitize_params(self, weights, mus, sigmas):
        """Numerically safe mixture parameters used identically in loss and prediction."""
        n_components = weights.shape[1]
        weights = torch.nan_to_num(weights, nan=1.0 / n_components, posinf=1.0, neginf=0.0)
        weights = weights.clamp(min=1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        mus = torch.nan_to_num(mus, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        sigmas = torch.nan_to_num(sigmas, nan=1.0, posinf=10.0, neginf=0.1).clamp(0.01, 10)
        return weights, mus, sigmas

    def _normalize_time(self, t):
        if self.time_norm_mode == 'min_max':
            mu = self.network.mu
            sigma = self.network.sigma
            # Shift to [1, 2] so inverse_softplus stays well away from its singularity at 0.
            return (t - mu) / (sigma + 1e-8) + 1.0
        return t

    def forward(self, x):
        return self.network(x)

    def calculate_loss(self, x, t, e):
        model_device = next(self.parameters()).device
        if x.device != model_device: x = x.to(model_device)
        if t.device != model_device: t = t.to(model_device)
        if e.device != model_device: e = e.to(model_device)

        weights, mus, sigmas = self.forward(x)                                  # (B, K) x3
        weights, mus, sigmas = self._sanitize_params(weights, mus, sigmas)

        t = t.view(-1, 1).clamp(min=1e-6)
        t_norm = self._normalize_time(t)
        inv_softplus_t = safe_inverse_softplus(t_norm)                          # (B, 1)

        dist = torch.distributions.Normal(mus, sigmas)
        inv_softplus_t_expanded = inv_softplus_t.expand(-1, mus.shape[1])       # (B, K)

        log_pdf_Tstar = dist.log_prob(inv_softplus_t_expanded)                  # (B, K)
        cdf = dist.cdf(inv_softplus_t_expanded)                                 # (B, K)

        # Chain rule: d(inv_softplus(t_norm))/dt = inverse_softplus_grad(t_norm) * d(t_norm)/dt
        log_jacobian = torch.log(inverse_softplus_grad(t_norm))                 # (B, 1)
        if self.time_norm_mode == 'min_max':
            log_jacobian = log_jacobian - torch.log(self.network.sigma + 1e-8)

        log_pdf = log_pdf_Tstar + log_jacobian                                  # (B, K)
        log_surv = torch.log(1.0 - cdf + 1e-7)                                  # (B, K)

        # Condition the normalized MDN on raw t >= 0. With min_max normalization raw t=0
        # maps to z0=1 (the stabilizing shift); mass below z0 is outside the raw-time
        # support and must be truncated away.
        z0 = self._normalize_time(torch.zeros_like(t))
        inv_softplus_z0 = safe_inverse_softplus(z0).expand(-1, mus.shape[1])
        log_surv_z0 = torch.log(1.0 - dist.cdf(inv_softplus_z0) + 1e-7)

        log_w = torch.log(weights + 1e-7)
        log_lik_pdf  = torch.logsumexp(log_w + log_pdf,  dim=1, keepdim=True)
        log_lik_surv = torch.logsumexp(log_w + log_surv, dim=1, keepdim=True)
        log_lik_z0   = torch.logsumexp(log_w + log_surv_z0, dim=1, keepdim=True)
        log_lik_pdf  = log_lik_pdf  - log_lik_z0
        log_lik_surv = log_lik_surv - log_lik_z0

        e = e.view(-1, 1)
        return -torch.mean(e * log_lik_pdf + (1 - e) * log_lik_surv)

    def predict_survival_probability(self, x, t):
        """S(t|x) = 1 - CDF(inverse_softplus(t)); forces S=1 at t<=0."""
        self.eval()
        model_device = next(self.parameters()).device
        x = self._to_tensor(x).to(model_device)
        t = self._to_tensor(t).to(model_device)

        with torch.no_grad():
            if t.dim() == 1:
                t = t.view(1, -1)
            n_times = t.shape[1]
            t_original = t.clone()
            t_norm = self._normalize_time(t.clamp(min=1e-6))
            inv_softplus_t = safe_inverse_softplus(t_norm)                      # (1, T)

            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_prob = []
            for batch in loader:
                batch_x = batch[0].to(model_device)
                batch_size = batch_x.shape[0]

                weights, mus, sigmas = self.forward(batch_x)
                weights, mus, sigmas = self._sanitize_params(weights, mus, sigmas)
                n_components = weights.shape[1]

                mus_exp = mus.unsqueeze(1).expand(-1, n_times, -1)               # (B, T, K)
                sigmas_exp = sigmas.unsqueeze(1).expand(-1, n_times, -1)
                dist = torch.distributions.Normal(mus_exp, sigmas_exp)
                inv_t_exp = inv_softplus_t.unsqueeze(-1).expand(batch_size, -1, n_components)

                surv_components = 1.0 - dist.cdf(inv_t_exp)                      # (B, T, K)
                surv = torch.sum(weights.unsqueeze(1) * surv_components, dim=2)  # (B, T)

                # Left-truncation normalizer at the raw-t=0 image.
                z0 = self._normalize_time(torch.zeros(batch_size, 1, device=model_device))
                inv_z0 = safe_inverse_softplus(z0).expand(-1, n_components)
                dist_z0 = torch.distributions.Normal(mus, sigmas)
                surv_z0 = torch.sum(weights * (1.0 - dist_z0.cdf(inv_z0)), dim=1, keepdim=True)
                surv = surv / surv_z0.clamp(min=1e-7)

                zero_mask = (t_original <= 1e-6).expand(batch_size, -1)
                surv = torch.where(zero_mask, torch.ones_like(surv), surv)
                surv = surv.clamp(0.0, 1.0)
                all_prob.append(surv.cpu())
            return torch.cat(all_prob, dim=0)

    def predict_hazard(self, x, t):
        """h(t|x) = f(t|x) / S(t|x) with f(t) = f_T*(inv_sp(t)) * |d(inv_sp)/dt|."""
        self.eval()
        model_device = next(self.parameters()).device
        x = self._to_tensor(x).to(model_device)
        t = self._to_tensor(t).to(model_device)

        with torch.no_grad():
            if t.dim() == 1:
                t = t.view(1, -1)
            n_times = t.shape[1]
            t_norm = self._normalize_time(t.clamp(min=1e-6))
            inv_softplus_t = safe_inverse_softplus(t_norm)

            loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
            all_haz = []
            for batch in loader:
                batch_x = batch[0].to(model_device)
                batch_size = batch_x.shape[0]

                weights, mus, sigmas = self.forward(batch_x)
                weights, mus, sigmas = self._sanitize_params(weights, mus, sigmas)
                n_components = weights.shape[1]

                mus_exp = mus.unsqueeze(1).expand(-1, n_times, -1)               # (B, T, K)
                sigmas_exp = sigmas.unsqueeze(1).expand(-1, n_times, -1)
                dist = torch.distributions.Normal(mus_exp, sigmas_exp)
                inv_t_exp = inv_softplus_t.unsqueeze(-1).expand(batch_size, -1, n_components)

                log_jacobian = torch.log(inverse_softplus_grad(t_norm)).unsqueeze(-1)
                if self.time_norm_mode == 'min_max':
                    log_jacobian = log_jacobian - torch.log(self.network.sigma + 1e-8)
                log_pdf = dist.log_prob(inv_t_exp) + log_jacobian
                pdf_components = torch.exp(log_pdf)
                surv_components = 1.0 - dist.cdf(inv_t_exp)

                weights_exp = weights.unsqueeze(1)
                f_t = torch.sum(weights_exp * pdf_components, dim=2)             # (B, T)
                s_t = torch.sum(weights_exp * surv_components, dim=2)
                # The left-truncation normalizer cancels in f/S, so conditioning at raw t=0
                # does not change the implied hazard.
                hazard = f_t / (s_t + 1e-8)
                all_haz.append(hazard.cpu())
            return torch.cat(all_haz, dim=0)
