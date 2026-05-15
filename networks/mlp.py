import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import BaseMLP
from .embeddings import TimeNormalizer, SinusoidalTimeEmbedding

def _apply_output_activation(out, name):
    if name == 'softplus':
        return F.softplus(out)
    if name == 'relu':
        return F.relu(out)
    if name == 'sigmoid':
        return torch.sigmoid(out)
    if name == 'softmax':
        return torch.softmax(out, dim=1)
    return out


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64], activation=nn.Tanh, dropout=0.0,
                 batch_norm=False, output_dim=1,
                 output_activation='identity', output_bias=True):
        super().__init__()
        self.output_activation = output_activation
        self.backbone = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation,
                                dropout=dropout, batch_norm=batch_norm, output_dim=None)
        self.head = nn.Linear(self.backbone.output_dim, output_dim, bias=output_bias)

    def forward(self, x):
        return _apply_output_activation(self.head(self.backbone(x)), self.output_activation)

class TimeConcatenatedMLP(nn.Module):
    """f(x, t) = MLP(concat(x, Norm(t)))."""

    def __init__(self, input_dim, hidden_dims=[64, 64], activation=nn.Tanh, dropout=0.0,
                 batch_norm=False, mu=0.0, sigma=1.0, time_norm_mode='min_max',
                 output_dim=1, output_activation='softplus', output_bias=True):
        super().__init__()
        self.output_activation = output_activation
        self.t_norm = TimeNormalizer(mu=mu, sigma=sigma, time_norm_mode=time_norm_mode or 'min_max')
        self.net = BaseMLP(input_dim=input_dim + 1, hidden_dims=hidden_dims, activation=activation,
                           dropout=dropout, batch_norm=batch_norm, output_dim=output_dim)

        if not output_bias:
            last_layer = self.net.net[-1]
            if isinstance(last_layer, nn.Linear):
                self.net.net[-1] = nn.Linear(last_layer.in_features, last_layer.out_features, bias=False)

        # softplus(-5) ≈ 0.0067 keeps initial cumulative hazards in a sane range.
        if output_activation == 'softplus' and output_bias:
            last_layer = self.net.net[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                nn.init.constant_(last_layer.bias, -5.0)

    @property
    def mu(self): return self.t_norm.mu
    @mu.setter
    def mu(self, value): self.t_norm.mu = value
    @property
    def sigma(self): return self.t_norm.sigma
    @sigma.setter
    def sigma(self, value): self.t_norm.sigma = value

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        out = self.net(torch.cat([x, self.t_norm(t)], dim=-1))
        return _apply_output_activation(out, self.output_activation)

    def forward_split(self, x, t):
        """Repeat x to match t when t has shape (B*K, 1) — used during quadrature."""
        if t.dim() == 1:
            t = t.unsqueeze(1)
        K = t.shape[0] // x.shape[0]
        x_rep = x.unsqueeze(1).expand(x.shape[0], K, *x.size()[1:]).reshape(-1, *x.size()[1:])
        return self.forward(x_rep, t)

class MDNNet(nn.Module):
    """Mixture density head producing (weights, mus, sigmas), each (B, n_components)."""

    def __init__(self, input_dim, hidden_dims=[64, 64], n_components=5, dropout=0.0,
                 batch_norm=False, mu=0.0, sigma=1.0):
        super().__init__()
        self.register_buffer('_mu', torch.tensor(float(mu)))
        self.register_buffer('_sigma', torch.tensor(float(sigma)))

        self.backbone = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=nn.Tanh,
                                dropout=dropout, batch_norm=batch_norm, output_dim=None)
        in_dim = self.backbone.output_dim

        self.fc_weights = nn.Linear(in_dim, n_components)
        self.fc_mu = nn.Linear(in_dim, n_components)
        self.fc_sigma = nn.Linear(in_dim, n_components)

        # Spread initial means across [-3, 3] so components don't all collapse to 0.
        self.fc_mu.weight.data.fill_(0.0)
        self.fc_mu.bias.data = torch.linspace(-3, 3, n_components)

    @property
    def mu(self): return self._mu
    @mu.setter
    def mu(self, value):
        self._mu = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=self._mu.device)

    @property
    def sigma(self): return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=self._sigma.device)

    def forward(self, x):
        h = self.backbone(x)
        weights = torch.softmax(self.fc_weights(h.clamp(-1e3, 1e3)), dim=1)
        mus = self.fc_mu(h).clamp(-30, 30)
        # Pre-clamp to a positive range so softplus(sigma_pre) >= softplus(0.01) ~= 0.69
        # — keeps components from collapsing to a Dirac.
        sigmas = F.softplus(self.fc_sigma(h).clamp(min=0.01, max=100.0)) + 1e-6
        return weights, mus, sigmas


# Aliases
class NeuralHazard(TimeConcatenatedMLP):
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.0, batch_norm=False, mu=0.0, sigma=1.0, time_norm_mode='min_max'):
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm, mu=mu, sigma=sigma, time_norm_mode=time_norm_mode, output_dim=1, output_activation='softplus')

class CoxTimeNet(TimeConcatenatedMLP):
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.0, batch_norm=False, mu=0.0, sigma=1.0, time_norm_mode='min_max'):
        super().__init__(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm, mu=mu, sigma=sigma, time_norm_mode=time_norm_mode, output_dim=1, output_activation='identity', output_bias=False)

CoxNet = SimpleMLP
DeepHitNet = SimpleMLP
NnetSurvNet = SimpleMLP



class PHNet(nn.Module):
    """Proportional Hazards: h(t|x) = h_0(t) * r(x), with both factors made positive via softplus.

    The x-head is bias-free for PH identifiability; the t-head bias is initialized to -3 so the
    initial baseline hazard is ~0.05, in the middle of a useful range.
    """

    def __init__(self, input_dim, hidden_dims=[64, 64], t_hidden_dims=[32, 32],
                 activation=nn.Tanh, dropout=0.0, batch_norm=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max'):
        super().__init__()
        self.time_norm_mode = time_norm_mode
        self.t_norm = TimeNormalizer(mu=mu, sigma=sigma, time_norm_mode=time_norm_mode)

        self.x_backbone = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation,
                                  dropout=dropout, batch_norm=batch_norm, output_dim=None)
        self.x_head = nn.Linear(self.x_backbone.output_dim, 1, bias=False)
        nn.init.zeros_(self.x_head.weight)

        t_layers = []
        t_in_dim = 1
        for t_h_dim in t_hidden_dims:
            t_layers.append(nn.Linear(t_in_dim, t_h_dim))
            t_layers.append(activation())
            t_in_dim = t_h_dim
        t_layers.append(nn.Linear(t_in_dim, 1))
        self.t_net = nn.Sequential(*t_layers)
        nn.init.zeros_(self.t_net[-1].weight)
        nn.init.constant_(self.t_net[-1].bias, -3.0)

    @property
    def mu(self): return self.t_norm.mu
    @mu.setter
    def mu(self, value): self.t_norm.mu = value
    @property
    def sigma(self): return self.t_norm.sigma
    @sigma.setter
    def sigma(self, value): self.t_norm.sigma = value

    def risk_multiplier(self, x):
        return F.softplus(self.x_head(self.x_backbone(x)))

    def baseline_hazard(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return F.softplus(self.t_net(self.t_norm(t)))

    def forward(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.baseline_hazard(t) * self.risk_multiplier(x)

    def forward_split(self, x, t):
        """Quadrature-friendly path: x is (B, D), t is (B*K, 1); broadcast risk across K."""
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.shape[0] % x.shape[0] != 0:
            raise ValueError(f"t size {t.shape[0]} must be divisible by x batch size {x.shape[0]}")
        K = t.shape[0] // x.shape[0]
        risk = self.risk_multiplier(x)
        risk_rep = risk.unsqueeze(1).expand(-1, K, -1).reshape(-1, 1)
        return self.baseline_hazard(t) * risk_rep

