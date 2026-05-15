import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .embeddings import TimeNormalizer


class TimeConcatSurvivalWrapper(nn.Module):
    """Backbone-agnostic time conditioning by concatenation: out = Head(cat(backbone(x), norm(t))).

    forward(x, t) expects t aligned with x; forward_split(x, t) takes a flattened t of length B*K
    and runs the backbone only once for efficiency.
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
        # Head MLP
        head_hidden_dims: list = [64, 64],
        head_activation: Callable[[], nn.Module] = nn.Tanh,
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
        self.output_activation = output_activation

        # MLP head: input = features + 1 (time scalar)
        layers = []
        in_dim = self.feature_dim + 1
        for h_dim in head_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(head_activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, int(output_dim), bias=bool(output_bias)))
        self.head = nn.Sequential(*layers)

        # Initialize output layer for proper hazard scale
        last = self.head[-1]
        if isinstance(last, nn.Linear) and output_bias:
            nn.init.zeros_(last.weight)
            if output_activation == 'softplus':
                nn.init.constant_(last.bias, -5.0)
            elif output_activation in ('exp', 'identity'):
                nn.init.constant_(last.bias, -3.0)

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
        return y  # identity

    def encode(self, x):
        return self.feature_fn(self.backbone, x)

    def forward(self, x, t):
        h = self.encode(x)                              # (B, D)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_n = self.t_norm(t)                            # (B, 1)
        combined = torch.cat([h, t_n], dim=-1)          # (B, D+1)
        y = self.head(combined)                         # (B, Out)
        return self._apply_activation(y)

    def forward_split(self, x, t, K=None):
        """Efficient path: backbone once, concat for all B*K time points using expand."""
        h, y_static, v = self.precompute(x)                              # (B, D)
        return self.forward_nodes_from_cache(h, y_static, v, t, K=K)

    def precompute(self, x: torch.Tensor):
        """Precompute static backbone features for multiple time evaluations."""
        h = self.encode(x)
        return h, None, None

    def forward_nodes_from_cache(self, h: torch.Tensor, y_static: torch.Tensor, v: torch.Tensor, t: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
        """Efficient forward using precomputed static terms with zero-copy expansion."""
        B, D = h.shape
        if t.dim() == 1:
            if K is None: K = t.numel() // B
            t_bk = t.view(B, K, 1)
        elif t.dim() == 2 and t.size(1) == 1:
            if K is None: K = t.size(0) // B
            t_bk = t.view(B, K, 1)
        elif t.dim() == 2:
            if K is None: K = t.size(1)
            t_bk = t.unsqueeze(-1)
        elif t.dim() == 3:
            t_bk = t
            if K is None: K = t.size(1)
        else:
            raise ValueError(f"Unsupported t shape: {t.shape}")

        t_n = self.t_norm(t_bk.reshape(-1, 1).contiguous())                       # (B*K, 1)

        # Zero-copy expansion instead of repeat_interleave
        h_expanded = h.unsqueeze(1).expand(-1, K, -1).reshape(B * K, D) # (B*K, D)
        combined = torch.cat([h_expanded, t_n], dim=-1)      # (B*K, D+1)

        y = self.head(combined)                         # (B*K, Out)
        return self._apply_activation(y)


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------
from .backbone import BaseMLP, ResNetBackbone


class ConcatMLP(TimeConcatSurvivalWrapper):
    """Two-stage concat for tabular: BaseMLP(x) → cat(h, t) → head."""
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.0, batch_norm=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max',
                 output_dim=1, output_activation='softplus', output_bias=True,
                 head_hidden_dims=[64]):
        backbone = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm)
        super().__init__(backbone, feature_dim=backbone.output_dim,
                         output_dim=output_dim, output_activation=output_activation, output_bias=output_bias,
                         mu=mu, sigma=sigma, time_norm_mode=time_norm_mode,
                         head_hidden_dims=head_hidden_dims)


class ConcatResNet(TimeConcatSurvivalWrapper):
    """Two-stage concat for images: ResNet(x) → cat(h, t) → head."""
    def __init__(self, model_name='resnet18', input_channels=1, pretrained=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max',
                 output_dim=1, output_activation='softplus', output_bias=True,
                 head_hidden_dims=[128]):
        backbone = ResNetBackbone(model_name=model_name, input_channels=input_channels, pretrained=pretrained)
        super().__init__(backbone, feature_dim=backbone.output_dim,
                         output_dim=output_dim, output_activation=output_activation, output_bias=output_bias,
                         mu=mu, sigma=sigma, time_norm_mode=time_norm_mode,
                         head_hidden_dims=head_hidden_dims)
