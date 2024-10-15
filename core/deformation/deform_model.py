import torch
import torch.nn as nn
import torch.nn.functional as F

from .rigid_utils import exp_se3
from core.optim.optim_utils import get_expon_lr_func
from configs import TrainConfig


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(
        self,
        xyz_input_ch=None,
        pose_input_ch=63,
        D=4,
        W=64,
        multires=10,
        residual=False,
        is_6dof=False,
    ):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2] if residual else []

        if xyz_input_ch is None:
            self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        else:
            self.embed_fn = nn.Identity()

        self.input_ch = xyz_input_ch + pose_input_ch
        
        layers = [nn.Linear(self.input_ch, W)]
        for i in range(D - 1):
            if i not in self.skips:
                layers.append(nn.Linear(W, W))
            else:
                layers.append(nn.Linear(W + self.input_ch, W))
        
        self.layers = nn.ModuleList(layers)
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, body_pose):
        """
        Args:
            x: torch.Tensor, [N, 3] or [N, C]
            body_pose: torch.Tensor, [N, 63]
        
        Return:
            d_xyz: torch.Tensor, [N, 3] or [N, 4, 4] if is_6dof
            rotation: torch.Tensor, [N, 4]
            scaling: torch.Tensor, [N, 3]
        """
        p_emb = body_pose.expand(x.shape[0], -1)  # [N, 63]
        x_emb = self.embed_fn(x)  # [N, 63]
        h = torch.cat([x_emb, p_emb], dim=-1)

        for i, linear in enumerate(self.layers):
            # h = F.relu(linear(h), inplace=True)
            h = F.leaky_relu(linear(h), inplace=True)
            if i in self.skips:
                h = torch.cat([x_emb, p_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            delta_positions = exp_se3(screw_axis, theta)
        else:
            delta_positions = self.gaussian_warp(h)
        
        delta_scales = self.gaussian_scaling(h)
        delta_quaternions = self.gaussian_rotation(h)

        # if delta_scales.requires_grad:
        #     def _hook(grad):
        #         grad_ = grad.detach().clone()
        #         print('scale:', grad_.shape, grad_.min().item(), grad_.std().item(), grad_.max().item())
        #     delta_scales.register_hook(_hook)

        return delta_positions, delta_scales, delta_quaternions

    def get_optimizer(self, cfg:TrainConfig):
        return DeformOptimizer(model=self, cfg=cfg)


class DeformOptimizer:
    def __init__(self, model:DeformNetwork, cfg:TrainConfig):
        
        self.current_iteration = 0

        params = [{
            'params': list(model.parameters()),
            'lr': cfg.render.position_lr_init,
            "name": "deform",
        }]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        deform_lr_delay_mult = 0.01
        deform_lr_max_steps = cfg.optim.iters
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=cfg.render.position_lr_init,
            lr_final=cfg.render.position_lr_final,
            lr_delay_mult=deform_lr_delay_mult,
            max_steps=deform_lr_max_steps,
        )
    
    def step(self):
        self.optimizer.step()
        self.current_iteration += 1
        
    def zero_grad(self, set_to_none:bool=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def update_learning_rate(self, iteration:int=None, spatial_scale:float=1.0):
        if iteration is None:
            iteration = self.current_iteration
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr * spatial_scale
                return lr
