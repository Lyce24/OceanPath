"""
SSL projection and prediction heads.

Three components:

1. **Projector** — maps slide_embedding [B, E] → projected [B, proj_dim].
   Used by all methods. Architecture: MLP with BN (SimCLR/BYOL style) or
   without BN (VICReg style, where BN would interfere with variance term).

2. **Predictor** — maps projected [B, proj_dim] → predicted [B, proj_dim].
   Used by BYOL and JEPA only. Prevents collapse in asymmetric methods.
   Smaller than projector (bottleneck).

3. **EMANetwork** — wraps any nn.Module with exponential moving average
   parameter and buffer updates. Used by BYOL (target encoder), DINO (teacher),
   and JEPA (target encoder).

4. **DINOHead** — special projector for DINO that maps to prototype space
   with L2 normalization + weight-normalized last layer (no bias).
"""

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations

logger = logging.getLogger(__name__)


class Projector(nn.Module):
    def __init__(
        self, in_dim=512, hidden_dim=2048, out_dim=256, num_layers=3, use_bn=True, last_bn=False
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            is_last = i == num_layers - 1
            has_bn = (last_bn and use_bn) if is_last else use_bn
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=not has_bn))
            if is_last:
                if has_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
            else:
                if has_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Predictor(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=1024, out_dim=256, use_bn=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=True)])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim=512,
        hidden_dim=2048,
        bottleneck_dim=256,
        out_dim=4096,
        num_layers=3,
        use_bn=True,
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 2) + [bottleneck_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.last_layer = parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False), name="weight", dim=0
        )

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class EMANetwork(nn.Module):
    def __init__(self, source_model, initial_momentum=0.996, final_momentum=1.0, total_steps=1000):
        super().__init__()
        self.ema_model = copy.deepcopy(source_model)
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.total_steps = max(1, total_steps)
        self._step = 0
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, source_model):
        m = self._get_momentum()
        for ema_p, src_p in zip(self.ema_model.parameters(), source_model.parameters()):
            ema_p.data.mul_(m).add_(src_p.data, alpha=1 - m)
        for ema_b, src_b in zip(self.ema_model.buffers(), source_model.buffers()):
            ema_b.data.copy_(src_b.data)
        self._step += 1

    def _get_momentum(self):
        progress = min(self._step / self.total_steps, 1.0)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_momentum - (self.final_momentum - self.initial_momentum) * cos_decay

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    @property
    def current_momentum(self):
        return self._get_momentum()
