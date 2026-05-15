import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_standard
from models.base import SurvivalModel
from torch.utils.data import DataLoader, TensorDataset

class SODEN(SurvivalModel):
    def __init__(self, hazard_net, optimizer_class=torch.optim.AdamW, scheduler_class=None, 
                 optimizer_params=None, scheduler_params=None, random_seed=42,
                 batch_size=128, epochs=100, lr=1e-3,
                 time_norm_mode='none', use_adjoint=True):
        super().__init__(hazard_net, optimizer_class, scheduler_class, optimizer_params, scheduler_params, random_seed,
                         batch_size, epochs, lr, time_norm_mode)
        self.network = hazard_net
        self.use_adjoint = use_adjoint
        
    def _internal_transform(self, t):
        if self.time_norm_mode == 'log_std':
            return torch.log1p(t)
        return t

    def _internal_inverse_transform(self, u):
        if self.time_norm_mode == 'log_std':
            return torch.exp(u) - 1.0
        return u

    def _jacobian(self, t):
        if self.time_norm_mode == 'log_std':
            return 1.0 / (1.0 + t)
        return torch.ones_like(t)

    def integrate_cumulative_hazard(self, x, t):
        """Integrate hazard from 0 to t in normalized v-space so the ODE solver stays well-conditioned."""
        x = self._to_tensor(x)
        t = self._to_tensor(t)
        if t.dim() == 1:
            t = t.view(-1, 1)

        mu = getattr(self.network, 'mu', 0.0)
        sigma = getattr(self.network, 'sigma', 1.0)
        if not isinstance(mu, torch.Tensor): mu = torch.tensor(mu).to(t.device)
        if not isinstance(sigma, torch.Tensor): sigma = torch.tensor(sigma).to(t.device)

        v0 = (torch.zeros_like(t) - mu) / (sigma + 1e-8)
        vT = (t - mu) / (sigma + 1e-8)
        diff_v = vT - v0

        def ode_func(tau, y):
            # tau in [0, 1]; map back to raw t for the network's internal time normalizer.
            t_eval = (v0 + tau * diff_v) * sigma + mu
            if t_eval.shape[0] == 1 and x.shape[0] > 1:
                t_eval = t_eval.expand(x.shape[0], -1)
            return self.network(x, t_eval) * diff_v * sigma

        t_span = torch.tensor([0., 1.], dtype=torch.float32, device=x.device)
        y0 = torch.zeros(x.shape[0], 1, dtype=torch.float32, device=x.device)

        if self.use_adjoint:
            sol = odeint_adjoint(ode_func, y0, t_span, method='dopri5', rtol=1e-4, atol=1e-8,
                                 adjoint_params=tuple(self.network.parameters()))
        else:
            sol = odeint_standard(ode_func, y0, t_span, method='dopri5', rtol=1e-4, atol=1e-8)
        return sol[-1]

    def calculate_loss(self, x, t, e):
        """NLL: -mean(e * log(h(t)) - H(t)); h(t) = h_net(u) * du/dt."""
        if t.ndim == 1:
            t = t.view(-1, 1)
        cum_hazard = self.integrate_cumulative_hazard(x, t)
        hazard = self.network(x, t) * self._jacobian(t)
        return -torch.mean(e.view(-1, 1) * torch.log(hazard + 1e-7) - cum_hazard)
    
    def predict_hazard(self, X, t):
        self.eval()
        X = self._to_tensor(X)
        t = self._to_tensor(t)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        n_times = t.shape[0]

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)
        all_haz = []
        jacob = self._jacobian(t).view(1, n_times)

        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(next(self.parameters()).device)
                b_size = batch_X.shape[0]

                if hasattr(self.network, 'forward_split'):
                    t_rep = t.unsqueeze(0).expand(b_size, -1).reshape(-1, 1).to(X.device)
                    h_net = self.network.forward_split(batch_X, t_rep)
                else:
                    batch_X_expanded = batch_X.unsqueeze(1).expand(batch_X.size(0), n_times, *batch_X.size()[1:]).reshape(-1, *batch_X.size()[1:])
                    batch_t_expanded = t.unsqueeze(0).expand(b_size, n_times).reshape(-1, 1).to(X.device)
                    h_net = self.network(batch_X_expanded, batch_t_expanded)

                hazard = h_net.reshape(b_size, n_times) * jacob.to(h_net.device)
                all_haz.append(hazard.cpu())

        return torch.cat(all_haz, dim=0)

    def predict_survival_probability(self, x, t):
        self.eval()
        x = self._to_tensor(x)
        t = self._to_tensor(t)

        if t.dim() != 1:
            raise NotImplementedError("Per-sample times (2D t) are not supported.")

        mu = getattr(self.network, 'mu', 0.0)
        sigma = getattr(self.network, 'sigma', 1.0)
        if not isinstance(mu, torch.Tensor): mu = torch.tensor(mu).to(t.device)
        if not isinstance(sigma, torch.Tensor): sigma = torch.tensor(sigma).to(t.device)

        v_times = (t - mu) / (sigma + 1e-8)
        v0 = (torch.zeros(1, device=t.device) - mu) / (sigma + 1e-8)
        eval_times = torch.unique(torch.cat([v0, v_times]), sorted=True)

        def ode_func_batch(tau, y, batch_x):
            t_eval = (tau * sigma + mu).expand(batch_x.shape[0], 1)
            return self.network(batch_x, t_eval) * sigma

        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
        all_surv = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(t.device)
                y0 = torch.zeros(batch_x.shape[0], 1).to(t.device)
                func = lambda time, y: ode_func_batch(time, y, batch_x)

                solver = odeint_adjoint if self.use_adjoint else odeint_standard
                kwargs = {'method': 'dopri5', 'rtol': 1e-4, 'atol': 1e-8}
                if self.use_adjoint:
                    kwargs['adjoint_params'] = ()
                sol = solver(func, y0, eval_times, **kwargs)                     # (len(eval_times), B, 1)

                indices = torch.searchsorted(eval_times, v_times)
                H_vals = sol[indices, :, 0].t()                                  # (B, T)
                all_surv.append(torch.exp(-H_vals).cpu())

        return torch.cat(all_surv, dim=0)
