import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from .embeddings import TimeNormalizer

def default_feature_fn(backbone, x):
    return backbone(x)


class TimeLoRAFeatureAdapter(nn.Module):
    """h' = h + scale * U(g(t) * V(h)); optionally also emits a per-time bias of shape (..., out_dim)."""
    def __init__(
        self,
        feature_dim: int,
        out_dim: int = 1,
        rank: int = 16,
        alpha: float = 16.0,
        t_hidden: int = 32,
        t_depth: int = 2,
        use_out_bias: bool = True,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.D = int(feature_dim)
        self.r = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)

        self.V = nn.Linear(self.D, self.r, bias=False)
        self.U = nn.Linear(self.r, self.D, bias=False)
        nn.init.zeros_(self.U.weight)

        self.use_out_bias = bool(use_out_bias)
        self.out_dim = int(out_dim)

        out_dim_total = self.r + (self.out_dim if self.use_out_bias else 0)

        layers = []
        in_dim = 1
        for _ in range(int(t_depth) - 1):
            layers += [nn.Linear(in_dim, int(t_hidden)), activation()]
            in_dim = int(t_hidden)
        final_layer = nn.Linear(in_dim, out_dim_total)

        # U is initialized to zero, so g must NOT also start at zero — otherwise the gradient
        # through delta = U(g*v) is identically zero. Initialize the g bias slice to 1.
        with torch.no_grad():
            nn.init.orthogonal_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
                final_layer.bias[:self.r].fill_(1.0)

        layers += [final_layer]
        self.t_mlp = nn.Sequential(*layers)

    def time_params(self, t_norm: torch.Tensor):
        params = self.t_mlp(t_norm)
        g = params[..., : self.r]
        b = params[..., self.r:] if self.use_out_bias else None
        return g, b

    def forward_v(self, h: torch.Tensor):
        return self.V(h)

    def delta_from_v(self, v: torch.Tensor, g: torch.Tensor):
        return self.U(g * v) * self.scale


class TimeLoRASurvivalWrapper(nn.Module):
    """Backbone-agnostic time conditioning via LoRA on pooled features.

    forward(x, t) expects aligned t; forward_split(x, t) takes a flattened t of length B*K and
    runs the backbone once while evaluating LoRA at every time point in parallel.
    """
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: Optional[int] = None,
        feature_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        # output head
        output_dim: int = 1,
        output_activation: str = "softplus",  # softplus for hazard output
        output_bias: bool = True,
        # time normalization
        mu: float = 0.0,
        sigma: float = 1.0,
        time_norm_mode: str = "min_max",  # Updated default
        # LoRA
        rank: int = 16,
        alpha: float = 32.0,
        t_hidden: int = 32,
        t_depth: int = 1,
        use_time_bias: bool = True,
        adapter_activation: Callable[[], nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_fn if feature_fn is not None else default_feature_fn
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # infer feature_dim if possible
        if feature_dim is None:
            if hasattr(backbone, "output_dim"):
                feature_dim = int(getattr(backbone, "output_dim"))
            elif hasattr(backbone, "fc") and isinstance(getattr(backbone, "fc"), nn.Linear):
                feature_dim = int(getattr(backbone, "fc").in_features)
            else:
                raise ValueError("Pass feature_dim explicitly (or provide feature_fn).")
        self.feature_dim = int(feature_dim)

        self.t_norm = TimeNormalizer(mu=mu, sigma=sigma, time_norm_mode=time_norm_mode)

        self.adapter = TimeLoRAFeatureAdapter(
            feature_dim=self.feature_dim,
            out_dim=output_dim,
            rank=rank,
            alpha=alpha,
            t_hidden=t_hidden,
            t_depth=t_depth,
            use_out_bias=use_time_bias,
            activation=adapter_activation,
        )

        self.head = nn.Linear(self.feature_dim, int(output_dim), bias=bool(output_bias))
        self.output_activation = output_activation

        # Bias init keeps initial hazards in a sane range: softplus(-5) ~ 0.0067, exp(-3) ~ 0.05.
        if output_bias:
            nn.init.zeros_(self.head.weight)
            if output_activation == 'softplus':
                nn.init.constant_(self.head.bias, -5.0)
            elif output_activation in ('exp', 'identity'):
                nn.init.constant_(self.head.bias, -3.0)

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

    def _apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "exp":
            return torch.exp(y.clamp(max=10))
        if self.output_activation == "softplus":
            return F.softplus(y)
        if self.output_activation == "relu":
            return F.relu(y)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(y)
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_fn(self.backbone, x)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.encode(x))
        if t.dim() == 1:
            t = t.unsqueeze(1)
        g, b_out = self.adapter.time_params(self.t_norm(t))
        v = self.adapter.forward_v(h)
        y = self.head(h + self.adapter.delta_from_v(v, g))
        if b_out is not None:
            y = y + b_out
        return self._apply_activation(y)

    def forward_split(self, x: torch.Tensor, t: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
        """Efficient path when t has length B*K — runs the backbone once."""
        h, y_static, v = self.precompute(x)
        return self.forward_nodes_from_cache(h, y_static, v, t, K=K)

    def precompute(self, x: torch.Tensor):
        h = self.dropout(self.encode(x))
        y_static = self.head(h)
        v = self.adapter.forward_v(h)
        return h, y_static, v

    def forward_nodes_from_cache(self, h: torch.Tensor, y_static: torch.Tensor, v: torch.Tensor,
                                 t: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
        B = y_static.shape[0]
        Out = self.head.out_features
        r = self.adapter.r

        if t.dim() == 1:
            K = K or (t.numel() // B)
            t_bk = t.view(B, K, 1)
        elif t.dim() == 2 and t.size(1) == 1:
            K = K or (t.shape[0] // B)
            t_bk = t.view(B, K, 1)
        elif t.dim() == 2:
            K = K or t.shape[1]
            t_bk = t.unsqueeze(-1)
        elif t.dim() == 3:
            K = K or t.shape[1]
            t_bk = t
        else:
            raise ValueError(f"Invalid t shape {t.shape}")

        # .contiguous() is needed for MPS stability (Apple GPU stride bug surfaces here).
        t_flat = t_bk.reshape(-1, 1).contiguous()
        g, b_out = self.adapter.time_params(self.t_norm(t_flat))
        g = g.view(B, K, r).contiguous()

        z = g * v.view(B, 1, r)
        W_eff = self.head.weight @ self.adapter.U.weight
        y_lora = (F.linear(z.reshape(B * K, r), W_eff) * self.adapter.scale).view(B, K, Out)
        y = y_static[:, None, :] + y_lora
        if b_out is not None:
            y = y + b_out.view(B, K, Out)
        return self._apply_activation(y.reshape(B * K, Out))

# -----------------------------------------------------------------------------
# Convenience Wrappers
# -----------------------------------------------------------------------------
from .backbone import BaseMLP, ResNetBackbone

class LoRAMLP(TimeLoRASurvivalWrapper):
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.0, batch_norm=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max', 
                 output_dim=1, output_activation='softplus', output_bias=True,
                 rank=32, alpha=32.0, t_hidden=32, t_depth=1):
        backbone = BaseMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, batch_norm=batch_norm)
        super().__init__(backbone, feature_dim=backbone.output_dim, output_dim=output_dim, output_activation=output_activation, output_bias=output_bias,
                         mu=mu, sigma=sigma, time_norm_mode=time_norm_mode,
                         rank=rank, alpha=alpha, t_hidden=t_hidden, t_depth=t_depth)

class LoRAResNet(TimeLoRASurvivalWrapper):
    def __init__(self, model_name='resnet18', input_channels=1, pretrained=False,
                 mu=0.0, sigma=1.0, time_norm_mode='min_max',
                 output_dim=1, output_activation='softplus', output_bias=True,
                 rank=32, alpha=32.0, t_hidden=128, t_depth=1, dropout=0.0):
        backbone = ResNetBackbone(model_name=model_name, input_channels=input_channels, pretrained=pretrained)
        super().__init__(backbone, feature_dim=backbone.output_dim, output_dim=output_dim, output_activation=output_activation, output_bias=output_bias,
                         mu=mu, sigma=sigma, time_norm_mode=time_norm_mode,
                         rank=rank, alpha=alpha, t_hidden=t_hidden, t_depth=t_depth, dropout=dropout)


