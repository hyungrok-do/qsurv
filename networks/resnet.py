import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNetBackbone
from .embeddings import TimeNormalizer
from .mlp import _apply_output_activation


class SimpleResNet(nn.Module):
    def __init__(self, model_name='resnet18', input_channels=1, output_dim=1,
                 output_activation='identity', output_bias=True, pretrained=False):
        super().__init__()
        self.output_activation = output_activation
        self.backbone = ResNetBackbone(model_name=model_name, input_channels=input_channels, pretrained=pretrained)
        self.head = nn.Linear(self.backbone.output_dim, output_dim, bias=output_bias)

    def forward(self, x):
        return _apply_output_activation(self.head(self.backbone(x)), self.output_activation)


class ResNetMDN(nn.Module):
    """ResNet feature extractor + mixture density head; sigma init mirrors MDNNet."""

    def __init__(self, model_name='resnet18', input_channels=1, n_components=5, pretrained=False):
        super().__init__()
        self.backbone = ResNetBackbone(model_name=model_name, input_channels=input_channels, pretrained=pretrained)
        in_dim = self.backbone.output_dim

        self.fc_weights = nn.Linear(in_dim, n_components)
        self.fc_mu = nn.Linear(in_dim, n_components)
        self.fc_sigma = nn.Linear(in_dim, n_components)

        self.fc_mu.weight.data.fill_(0.0)
        self.fc_mu.bias.data = torch.linspace(-3, 3, n_components)

    def forward(self, x):
        h = self.backbone(x).clamp(-1e3, 1e3)
        weights = torch.softmax(self.fc_weights(h).clamp(-50, 50), dim=1)
        mus = self.fc_mu(h).clamp(-30, 30)
        sigmas = F.softplus(self.fc_sigma(h).clamp(min=0.01, max=100.0)) + 1e-6
        return weights, mus, sigmas


class ResNetCox(SimpleResNet):
    def __init__(self, model_name='resnet18', input_channels=1, pretrained=False):
        super().__init__(model_name=model_name, input_channels=input_channels,
                         output_dim=1, output_activation='identity', output_bias=False, pretrained=pretrained)


class ResNetDeepHit(SimpleResNet):
    def __init__(self, output_dim, model_name='resnet18', input_channels=1, pretrained=False):
        super().__init__(model_name=model_name, input_channels=input_channels,
                         output_dim=output_dim, output_activation='identity', pretrained=pretrained)


class ResNetNnetSurv(SimpleResNet):
    def __init__(self, output_dim, model_name='resnet18', input_channels=1, pretrained=False):
        super().__init__(model_name=model_name, input_channels=input_channels,
                         output_dim=output_dim, output_activation='identity', pretrained=pretrained)


class ResNetPHNet(nn.Module):
    """Image PH model: h(t|x) = h_0(t) * r(x); same softplus / no-bias / init choices as PHNet."""

    def __init__(self, model_name='resnet18', input_channels=1, t_hidden_dims=[32, 32],
                 pretrained=False, mu=0.0, sigma=1.0, time_norm_mode='min_max'):
        super().__init__()
        self.time_norm_mode = time_norm_mode
        self.t_norm = TimeNormalizer(mu=mu, sigma=sigma, time_norm_mode=time_norm_mode)

        self.backbone = ResNetBackbone(model_name=model_name, input_channels=input_channels, pretrained=pretrained)
        self.x_head = nn.Linear(self.backbone.output_dim, 1, bias=False)
        nn.init.zeros_(self.x_head.weight)

        t_layers = []
        t_in_dim = 1
        for t_h_dim in t_hidden_dims:
            t_layers.append(nn.Linear(t_in_dim, t_h_dim))
            t_layers.append(nn.Tanh())
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
        return F.softplus(self.x_head(self.backbone(x)))

    def baseline_hazard(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return F.softplus(self.t_net(self.t_norm(t)))

    def forward(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.baseline_hazard(t) * self.risk_multiplier(x)

    def forward_split(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.shape[0] % x.shape[0] != 0:
            raise ValueError(f"t size {t.shape[0]} must be divisible by x batch size {x.shape[0]}")
        K = t.shape[0] // x.shape[0]
        risk = self.risk_multiplier(x)
        risk_rep = risk.unsqueeze(1).expand(-1, K, -1).reshape(-1, 1)
        return self.baseline_hazard(t) * risk_rep
