import torch
import torch.nn as nn
import torchvision.models as models


class BaseMLP(nn.Module):
    """Multi-Layer Perceptron with optional final Linear projection."""

    def __init__(self, input_dim, hidden_dims=[64, 64], activation=nn.Tanh, dropout=0.0,
                 batch_norm=False, output_dim=None):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim if output_dim is None else output_dim

    def forward(self, x):
        return self.net(x)


class ResNetBackbone(nn.Module):
    """torchvision ResNet with the FC head removed; first conv re-shaped for non-RGB input."""

    def __init__(self, model_name='resnet18', input_channels=1, pretrained=False):
        super().__init__()
        try:
            model_fn = getattr(models, model_name)
        except AttributeError:
            raise ValueError(f"Unknown model name: {model_name}")

        weights = None
        if pretrained:
            weight_enum_name = f"{model_name.replace('resnet', 'ResNet')}_Weights"
            try:
                weight_class = getattr(models, weight_enum_name, None)
                weights = weight_class.IMAGENET1K_V1 if weight_class else 'IMAGENET1K_V1'
            except Exception:
                weights = None

        try:
            self.resnet = model_fn(weights=weights)
        except TypeError:
            self.resnet = model_fn(pretrained=pretrained)

        if input_channels != 3:
            old_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                input_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=(old_conv.bias is not None),
            )
            if pretrained and input_channels == 1:
                # Average across the 3 RGB filters to seed the grayscale filter.
                with torch.no_grad():
                    self.resnet.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.output_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)
