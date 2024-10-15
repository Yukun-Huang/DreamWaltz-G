import torch
from typing import Tuple, Mapping, Any, Optional
from dataclasses import dataclass, asdict, fields

from .spherical_harmonics import eval_sh


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def get_colors(sh_features, directions, sh_levels):
    sh_features = sh_features[:, :sh_levels**2]
    shs_view = sh_features.transpose(-1, -2).view(-1, 3, sh_levels**2)
    sh2rgb = eval_sh(sh_levels-1, shs_view, directions)
    colors = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
    return colors


@dataclass
class GaussianOutput:
    positions: Optional[torch.Tensor] = None
    sh_features: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    quaternions: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    # for precompute
    colors: Optional[torch.Tensor] = None
    cov3D: Optional[torch.Tensor] = None
    # for deformation
    offsets: Optional[torch.Tensor] = None
    lbs_weights: Optional[torch.Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


def merge_gaussians(*gaussians: GaussianOutput) -> GaussianOutput:
    if len(gaussians) == 1:
        return gaussians[0]
    new_gaussians = asdict(GaussianOutput())
    for k in new_gaussians.keys():
        all_params = []
        for gaussian in gaussians:
            params = gaussian[k]
            if torch.is_tensor(params):
                all_params.append(params)
        if len(all_params) > 0:
            new_gaussians[k] = torch.cat(all_params, dim=0)
    return GaussianOutput(**new_gaussians)


def downsample_gaussians(gaussians: GaussianOutput, n_gaussians: int) -> GaussianOutput:
    new_gaussians = asdict(GaussianOutput())
    n_points = gaussians.positions.size(0)
    indices = torch.randperm(n_points)[:n_gaussians]
    for k in new_gaussians.keys():
        params = gaussians[k]
        if torch.is_tensor(params):
            new_gaussians[k] = params[indices]
        else:
            new_gaussians[k] = params
    return GaussianOutput(**new_gaussians)
