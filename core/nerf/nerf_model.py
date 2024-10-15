import torch
from torch import nn
import torch.nn.functional as F
from random import random
from typing import Optional
from configs import NeRFConfig
from .encoding import get_encoder
from .nerf_utils import trunc_exp, safe_normalize, init_decoder_layer
from .nerf_renderer import NeRFRenderer


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            in_features = self.dim_in if l == 0 else self.dim_hidden
            out_features = self.dim_out if l == num_layers - 1 else self.dim_hidden
            net.append(nn.Linear(in_features, out_features, bias=bias))
        
        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class BaseNeRFNetwork(NeRFRenderer):
    # add a density blob to the scene center
    def density_prior(self, x):
        # x: [B, N, 3]
        if self.density_prior_type == 'none':
            return 0.0
        elif self.density_prior_type == 'smpl':
            return self.density_smpl(x)
        d = (x ** 2).sum(-1)
        if self.density_prior_type == 'gaussian':
            gaussian_peak, gaussian_std = 5, 0.2
            g = gaussian_peak * torch.exp(-d / (2 * gaussian_std ** 2))
        elif self.density_prior_type == 'sqrt':
            blob_density, blob_radius = 10, 0.5
            g = blob_density * (1 - torch.sqrt(d) / blob_radius)
        else:
            assert 0, self.density_prior_type
        return g

    def postprocess(self, inputs):
        if self.latent_mode:
            if self.decoder_layer is not None:
                inputs = self.decoder_layer(inputs)
            return inputs
        else:
            if self.decoder_layer is not None:
                inputs = self.decoder_layer(inputs)
                # inputs = (inputs + 1) / 2
            return torch.sigmoid(inputs)

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        sigma, albedo = self.common_forward(x)
        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        # if x.shape[0] == 0:
        #     assert 0, f'Found empty ray points! {x.shape}'
        
        sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            color = albedo
        else:
            normal = self.normal(x)
            if shading == 'normal':
                # normal shading
                color = (normal + 1.0) / 2.0
            else:
                # lambertian shading
                lambertian = ratio + (1 - ratio) * (normal @ -l).clamp(min=0)  # [N,]
                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, 3)
                elif shading == 'lambertian':
                    color = albedo * lambertian.unsqueeze(-1)
                else:
                    assert 0, shading

            if self.latent_mode:
                # pad color with a single dimension of zeros
                color = torch.cat([color, torch.zeros((color.shape[0], 1), device=color.device)], axis=1)

        return sigma, color

    def background(self, image:torch.Tensor, bg_mode:Optional[str], rays_d:torch.Tensor):
        """
          - image: torch.Tensor [B, H, W, 3]
          - weights_sum: torch.Tensor [B, H, W, 1]
          - bg_mode: str
          - rays_d: torch.Tensor [B, H*W, 3]
        """
        if bg_mode is None:
            bg_mode = self.bg_mode
        
        # use random background color with probability random_aug_prob
        if self.training and self.rand_bg_prob is not None:
            if random() < self.rand_bg_prob:
                bg_mode = 'gray'

        if bg_mode in ('none', None, 'disable'):
            return None

        elif bg_mode in ('zero', 'zeros'):
            bg_image = torch.zeros_like(image)
        
        elif bg_mode == 'normal':
            bg_image = torch.randn_like(image)
        
        elif bg_mode == 'uniform':
            bg_image = torch.rand(self.img_dims).to(image).expand_as(image)
        
        elif bg_mode == 'nerf':
            d = rays_d.contiguous().view(-1, 3)
            h = self.encoder_bg(d)  # [N, C]
            h = self.bg_net(h)
            h = self.postprocess(h)
            bg_image = h.reshape_as(image)
        
        else:
            bg_image = self.bg_colors[bg_mode].to(image).expand_as(image)

        return bg_image

    def normal(self, x, normal_type:str='finite_difference_laplacian', epsilon:float=1e-3):
        # x: [N, 3]
        if normal_type == 'finite_difference_laplacian':
            dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
            dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
            normal = - 0.5 * torch.stack([
                (dx_pos - dx_neg),
                (dy_pos - dy_neg),
                (dz_pos - dz_neg)
            ], dim=-1) / epsilon
        else:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x.requires_grad_(True)
                    sigma, albedo = self.common_forward(x)
                    # query gradient
                    normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal

    def build_optimizer(self, params, optimizer_type):
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params, betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-3)  # weight_decay=1e-2
        elif optimizer_type == 'adan':
            from core.optim.adan import Adan
            optimizer = Adan(params, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        elif optimizer_type == 'adamax':
            optimizer = torch.optim.Adamax(params, betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params)
        else:
            raise NotImplementedError
        return optimizer

    def build_scheduler(self, optimizer, lr_policy, diffusion=None):
        from core.optim.scheduler import make_scheduler
        if lr_policy in ('none', 'constant'):
            scheduler = None
        elif lr_policy == 'step':
            scheduler = make_scheduler(optimizer, lr_policy, step_size=int(self.max_step * 0.7))
        elif lr_policy == 'multistep':
            scheduler = make_scheduler(optimizer, lr_policy, step_size=int(self.max_step * 0.7))
        elif lr_policy == 'warmup':
            scheduler = make_scheduler(optimizer, lr_policy, milestones=[int(self.max_step * 0.7),], warmup_iter=1000)
        elif lr_policy == 'lambda':
            def lr_lambda(i: int):
                # i: 0, 1, ..., 10000
                idx = int((1 - i / lr_lambda.max_step) * 1000)  # 1000 -> 0
                if idx == 1000:
                    return 1.0
                else:
                    return 1 - lr_lambda.alphas_cumprod[idx].item()
            lr_lambda.max_step = self.max_step
            lr_lambda.alphas_cumprod = diffusion.alphas_cumprod  # [0.9991 -> 0.0047]
            scheduler = make_scheduler(optimizer, lr_policy, lr_lambda=lr_lambda)
        elif lr_policy == 'ddpm':
            from core.guidance.time_prior import TimePrioritizedLR
            scheduler = TimePrioritizedLR(diffusion.tp_scheduler, optimizer=optimizer)
        return scheduler


class _NeRFNetwork(BaseNeRFNetwork):
    def __init__(self, cfg: NeRFConfig, num_layers=3, hidden_dim=64, num_layers_bg=2, hidden_dim_bg=64):
        super().__init__(cfg)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        additional_dim_size = self.additional_dim_size
        density_activation = cfg.density_activation

        # Foreground Network
        self.encoder, self.in_dim = get_encoder(
            cfg.backbone,
            input_dim=3,
            desired_resolution=cfg.desired_resolution * self.bound,
            num_levels=cfg.num_levels,
            level_dim=cfg.level_dim,
            base_resolution=cfg.base_resolution,
            interpolation='smoothstep',
        )
        self.sigma_net = MLP(self.in_dim, 4 + additional_dim_size, hidden_dim, num_layers, bias=True)
        self.sigma_scale = torch.nn.Parameter(torch.tensor(0.0))

        if density_activation == 'exp':
            self.density_activation = trunc_exp
        elif density_activation == 'softplus':
            self.density_activation = F.softplus
        elif density_activation == 'scaling':
            def density_activation(x, density_shift=-1.0):
                x = x * torch.exp(self.sigma_scale)
                return F.softplus(x + density_shift)
            self.density_activation = density_activation
        else:
            assert 0
        self.density_prior_type = cfg.density_prior

        # Background Network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)
            self.bg_net = MLP(self.in_dim_bg, 3 + additional_dim_size, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None

        # Latent-NeRF-Decoder-Layer
        if cfg.nerf_type == 'latent_tune':
            self.decoder_layer = nn.Linear(4, 3, bias=False)
            init_decoder_layer(self.decoder_layer)
        elif cfg.nerf_type == 'latent_approx':
            self.decoder_layer = nn.Linear(3, 4, bias=False)
            init_decoder_layer(self.decoder_layer, inverse=True)
        else:
            self.decoder_layer = None
    
    def local_geometry_forward(self, x, mlp_no_grad:bool=False):
        enc = self.encoder(x, bound=self.bound)
        if mlp_no_grad:
            requires_grad = next(self.sigma_net.parameters()).requires_grad
            self.sigma_net.requires_grad_(False)
        h = self.sigma_net(enc)
        sigma, albedo = h[..., 0], h[..., 1:]
        albedo = self.postprocess(albedo)
        if mlp_no_grad:
            self.sigma_net.requires_grad_(requires_grad)
        return sigma, albedo

    def common_forward(self, x, mask=None, return_raw=False, **kwargs):
        # forward
        enc = self.encoder(x, bound=self.bound)
        h = self.sigma_net(enc)
        sigma, albedo = h[..., 0], h[..., 1:]
        # return without activation
        if return_raw:
            return sigma, albedo
        # sigma
        sigma = self.density_activation(sigma + self.density_prior(x))
        if mask is not None:
            sigma *= mask
        # albedo
        albedo = self.postprocess(albedo)
        # return
        return sigma, albedo

    def get_optimizer(self, cfg: NeRFConfig, diffusion=None):

        lr = cfg.lr
        bg_lr = cfg.bg_lr

        if cfg.optimizer == 'adan':
            lr *= 5
            bg_lr *= 5

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.sigma_scale, 'lr': lr},
        ]

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': bg_lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': bg_lr})

        if self.decoder_layer is not None:
            params.append({'params': self.decoder_layer.parameters(), 'lr': lr})

        if self.dmtet and not self.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})
        
        # Optimizer
        optimizer = self.build_optimizer(params, optimizer_type=cfg.optimizer)

        # LR Scheduler
        scheduler = self.build_scheduler(optimizer, lr_policy=cfg.lr_policy, diffusion=diffusion)

        return optimizer, scheduler


class _NeRFNetwork_DualMLP(BaseNeRFNetwork):
    def __init__(self, cfg: NeRFConfig, num_layers=3, hidden_dim=64, num_layers_bg=2, hidden_dim_bg=64):

        super().__init__(cfg)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        additional_dim_size = self.additional_dim_size
        density_activation = cfg.density_activation
        
        # Foreground Network
        self.encoder, self.in_dim = get_encoder(
            cfg.backbone,
            input_dim=3,
            desired_resolution=cfg.desired_resolution * self.bound,
            num_levels=cfg.num_levels,
            level_dim=cfg.level_dim,
            base_resolution=cfg.base_resolution,
            interpolation='smoothstep',
        )
        self.albedo_net = MLP(self.in_dim, 3 + additional_dim_size, hidden_dim, num_layers, bias=True)
        self.sigma_net = MLP(self.in_dim, 1, hidden_dim, num_layers, bias=True)

        if density_activation == 'exp':
            self.density_activation = trunc_exp
        elif density_activation == 'softplus':
            self.density_activation = F.softplus
        elif density_activation == 'scaling':
            self.sigma_scale = torch.nn.Parameter(torch.tensor(0.0))
            def density_activation(x, density_shift=-1.0):
                x = x * torch.exp(self.sigma_scale)
                return F.softplus(x + density_shift)
            self.density_activation = density_activation
        else:
            assert 0
        self.density_prior_type = cfg.density_prior

        # Background Network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)
            self.bg_net = MLP(self.in_dim_bg, 3 + additional_dim_size, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None

        # Latent-NeRF-Decoder-Layer
        if cfg.nerf_type == 'latent_tune':
            self.decoder_layer = nn.Linear(4, 3, bias=False)
            init_decoder_layer(self.decoder_layer)
        else:
            self.decoder_layer = None

    def common_forward(self, x, mask=None, **kwargs):
        # forward
        enc = self.encoder(x, bound=self.bound)
        sigma = self.sigma_net(enc)[..., 0]
        albedo = self.albedo_net(enc)

        # sigma
        sigma = self.density_activation(sigma + self.density_prior(x))
        if mask is not None:
            sigma *= mask
        
        # albedo
        albedo = self.postprocess(albedo)
        
        # return
        return sigma, albedo

    def get_optimizer(self, cfg: NeRFConfig, diffusion=None):

        lr = cfg.lr
        bg_lr = cfg.bg_lr

        if cfg.optimizer == 'adan':
            lr *= 5
            bg_lr *= 5

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.albedo_net.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]

        if hasattr(self, 'sigma_scale'):
            params.append({'params': self.sigma_scale, 'lr': lr})

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': bg_lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': bg_lr})
        
        if self.decoder_layer is not None:
            params.append({'params': self.decoder_layer.parameters(), 'lr': lr})

        if self.dmtet and not self.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})
        
        # Optimizer
        optimizer = self.build_optimizer(params, optimizer_type=cfg.optimizer)

        # LR Scheduler
        scheduler = self.build_scheduler(optimizer, lr_policy=cfg.lr_policy, diffusion=diffusion)

        return optimizer, scheduler


class _NeRFNetwork_DualEnc(BaseNeRFNetwork):
    def __init__(self, cfg: NeRFConfig, num_layers=3, hidden_dim=64, num_layers_bg=2, hidden_dim_bg=64):

        super().__init__(cfg)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        additional_dim_size = self.additional_dim_size
        density_activation = cfg.density_activation

        # Foreground Network
        self.encoder, self.in_dim = get_encoder(
            cfg.backbone,
            input_dim=3,
            desired_resolution=cfg.desired_resolution * self.bound,
            num_levels=cfg.num_levels,
            level_dim=cfg.level_dim,
            base_resolution=cfg.base_resolution,
            interpolation='smoothstep',
        )
        self.encoder_sigma, self.in_dim_sigma = get_encoder(
            cfg.backbone,
            input_dim=3,
            desired_resolution=cfg.desired_resolution * self.bound,
            num_levels=cfg.num_levels,
            level_dim=cfg.level_dim,
            base_resolution=cfg.base_resolution,
            interpolation='smoothstep',
        )
        self.albedo_net = MLP(self.in_dim, 3 + additional_dim_size, hidden_dim, num_layers, bias=True)
        self.sigma_net = MLP(self.in_dim_sigma, 1, hidden_dim, num_layers, bias=True)

        if density_activation == 'exp':
            self.density_activation = trunc_exp
        elif density_activation == 'softplus':
            self.density_activation = F.softplus
        elif density_activation == 'scaling':
            self.sigma_scale = torch.nn.Parameter(torch.tensor(0.0))
            def density_activation(x, density_shift=-1.0):
                x = x * torch.exp(self.sigma_scale)
                return F.softplus(x + density_shift)
            self.density_activation = density_activation
        else:
            assert 0
        self.density_prior_type = cfg.density_prior

        # Background Network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)
            self.bg_net = MLP(self.in_dim_bg, 3 + additional_dim_size, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None

        # Latent-NeRF-Decoder-Layer
        if cfg.nerf_type == 'latent_tune':
            self.decoder_layer = nn.Linear(4, 3, bias=False)
            init_decoder_layer(self.decoder_layer)
        else:
            self.decoder_layer = None
    
    def common_forward(self, x, mask=None, **kwargs):
        # forward
        enc = self.encoder(x, bound=self.bound)
        albedo = self.albedo_net(enc)

        enc_sigma = self.encoder_sigma(x, bound=self.bound)
        sigma = self.sigma_net(enc_sigma)[..., 0]

        # sigma
        sigma = self.density_activation(sigma + self.density_prior(x))
        if mask is not None:
            sigma *= mask
        
        # albedo
        albedo = self.postprocess(albedo)
        
        # return
        return sigma, albedo

    def get_optimizer(self, cfg: NeRFConfig, diffusion=None):

        lr = cfg.lr
        bg_lr = cfg.bg_lr

        if cfg.optimizer == 'adan':
            lr *= 5
            bg_lr *= 5

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.encoder_sigma.parameters(), 'lr': lr * 10},
            {'params': self.albedo_net.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]

        if hasattr(self, 'sigma_scale'):
            params.append({'params': self.sigma_scale, 'lr': lr})

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': bg_lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': bg_lr})
        
        if self.decoder_layer is not None:
            params.append({'params': self.decoder_layer.parameters(), 'lr': lr})

        if self.dmtet and not self.lock_geo:
            params.append({'params': self.sdf, 'lr': lr})
            params.append({'params': self.deform, 'lr': lr})
        
        # Optimizer
        optimizer = self.build_optimizer(params, optimizer_type=cfg.optimizer)

        # LR Scheduler
        scheduler = self.build_scheduler(optimizer, lr_policy=cfg.lr_policy, diffusion=diffusion)

        return optimizer, scheduler


NeRFNetwork = _NeRFNetwork


def build_NeRFNetwork(cfg: NeRFConfig):
    structure = cfg.structure
    if structure == 'shared_mlp':
        return _NeRFNetwork(cfg)
    elif structure == 'dual_mlp':
        return _NeRFNetwork_DualMLP(cfg)
    elif structure == 'dual_enc':
        return _NeRFNetwork_DualEnc(cfg)
    else:
        assert 0, structure
