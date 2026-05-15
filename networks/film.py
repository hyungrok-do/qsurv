import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .embeddings import TimeNormalizer


class TimeFiLMSurvivalWrapper(nn.Module):
    """Backbone-agnostic time conditioning via FiLM: h' = (1 + gamma(t)) * h + beta(t).

    forward(x, t) expects aligned t; forward_split(x, t) takes a flattened t of length B*K and
    runs the backbone only once.
    """
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: Optional[int] = None,
        feature_fn: Optional[Callable] = None,
        # output head
        output_dim: int = 1,
        output_activation: str = "softplus",
        output_bias: bool = True,
        # time normalization
        mu: float = 0.0,
        sigma: float = 1.0,
        time_norm_mode: str = "min_max",
        # FiLM MLP
        t_hidden: int = 32,
        t_depth: int = 2,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_fn if feature_fn is not None else lambda bb, x: bb(x)

        # infer feature_dim
        if feature_dim is None:
            if hasattr(backbone, "output_dim"):
                feature_dim = int(getattr(backbone, "output_dim"))
            else:
                raise ValueError("Pass feature_dim explicitly.")
        self.feature_dim = int(feature_dim)

        self.t_norm = TimeNormalizer(mu=mu, sigma=sigma, time_norm_mode=time_norm_mode)

        # t -> (gamma, beta), each of size feature_dim
        layers = []
        in_dim = 1
        for _ in range(int(t_depth) - 1):
            layers += [nn.Linear(in_dim, int(t_hidden)), activation()]
            in_dim = int(t_hidden)
        layers.append(nn.Linear(in_dim, 2 * self.feature_dim))
        self.t_mlp = nn.Sequential(*layers)

        # Identity init: gamma = 1 (forward adds 1 to bias=0), beta = 0.
        with torch.no_grad():
            self.t_mlp[-1].weight.zero_()
            self.t_mlp[-1].bias.zero_()

        self.head = nn.Linear(self.feature_dim, int(output_dim), bias=bool(output_bias))
        self.output_activation = output_activation

        # Bias init keeps initial hazards in a sane range: softplus(-5) ~ 0.0067, exp(-3) ~ 0.05.
        if output_bias:
            nn.init.zeros_(self.head.weight)
            if output_activation == 'softplus':
                nn.init.constant_(self.head.bias, -5.0)
            elif output_activation in ('exp', 'identity'):
                nn.init.constant_(self.head.bias, -3.0)

    # -- Properties for QSurv time norm injection --
    @property
    def mu(self):
        return self.t_norm.mu

    @mu.setter
    def mu(self, value):
        self.t_norm.mu = value

    @property
    def sigma(self):
        return self.t_norm.sigma

    @sigma.setter
    def sigma(self, value):
        self.t_norm.sigma = value

    def _apply_activation(self, y):
        if self.output_activation == "exp":
            return torch.exp(y.clamp(max=10))
        if self.output_activation == "softplus":
            return F.softplus(y)
        if self.output_activation == "relu":
            return F.relu(y)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(y)
        return y

    def _film_transform(self, h, t_norm):
        params = self.t_mlp(t_norm)
        gamma = params[..., :self.feature_dim]
        beta = params[..., self.feature_dim:]
        return (1.0 + gamma) * h + beta

    def encode(self, x):
        return self.feature_fn(self.backbone, x)

    def forward(self, x, t):
        h = self.encode(x)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self._apply_activation(self.head(self._film_transform(h, self.t_norm(t))))

    def forward_split(self, x, t, K=None):
        h, y_static, v = self.precompute(x)
        return self.forward_nodes_from_cache(h, y_static, v, t, K=K)

    def precompute(self, x: torch.Tensor):
        h = self.encode(x)
        return h, h, None

    def forward_nodes_from_cache(self, h: torch.Tensor, y_static: torch.Tensor, v: torch.Tensor,
                                 t: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
        B, D = h.shape
        if t.dim() == 1:
            K = K or (t.numel() // B)
            t_bk = t.view(B, K, 1)
        elif t.dim() == 2 and t.size(1) == 1:
            K = K or (t.size(0) // B)
            t_bk = t.view(B, K, 1)
        elif t.dim() == 2:
            K = K or t.size(1)
            t_bk = t.unsqueeze(-1)
        elif t.dim() == 3:
            t_bk = t
            K = K or t.size(1)
        else:
            raise ValueError(f"Unsupported t shape: {t.shape}")

        t_n = self.t_norm(t_bk.reshape(-1, 1).contiguous()).view(B, K, 1)
        params = self.t_mlp(t_n)
        gamma = params[..., :self.feature_dim]
        beta = params[..., self.feature_dim:]
        h_prime = (1.0 + gamma) * h.view(B, 1, D) + beta                      # (B, K, D)
        return self._apply_activation(self.head(h_prime.reshape(B * K, D)))


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------
from .backbone import BaseMLP, ResNetBackbone


class FiLMMLP(TimeFiLMSurvivalWrapper):
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.0, batch_norm=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max',
                 output_dim=1, output_activation='softplus', output_bias=True,
                 t_hidden=32, t_depth=2):
        backbone = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm)
        super().__init__(backbone, feature_dim=backbone.output_dim,
                         output_dim=output_dim, output_activation=output_activation, output_bias=output_bias,
                         mu=mu, sigma=sigma, time_norm_mode=time_norm_mode,
                         t_hidden=t_hidden, t_depth=t_depth)


class FiLMResNet(TimeFiLMSurvivalWrapper):
    def __init__(self, model_name='resnet18', input_channels=1, pretrained=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max',
                 output_dim=1, output_activation='softplus', output_bias=True,
                 t_hidden=128, t_depth=2):
        backbone = ResNetBackbone(model_name=model_name, input_channels=input_channels, pretrained=pretrained)
        super().__init__(backbone, feature_dim=backbone.output_dim,
                         output_dim=output_dim, output_activation=output_activation, output_bias=output_bias,
                         mu=mu, sigma=sigma, time_norm_mode=time_norm_mode,
                         t_hidden=t_hidden, t_depth=t_depth)


