import torch
import torch.nn as nn
from typing import Mapping, Any, Iterable, Optional
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from configs import TrainConfig
from core.gaussian.gaussian_model import GaussianModel
from core.gaussian.gaussian_utils import GaussianOutput, merge_gaussians, downsample_gaussians
from core.gaussian.gaussian_renderer import GaussianRenderer
from core.system.avatar import Avatar
from core.system.background import MLPBackground, PureColorBackground, VideoBackground


class Scene(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        avatar: Avatar | Iterable[Avatar],
        background: Optional[GaussianModel | MLPBackground | VideoBackground] = None,
    ) -> None:
        super().__init__()
        
        self.device = torch.device(cfg.device)
        
        # Models
        if isinstance(avatar, Avatar):
            self.avatar = avatar
            self.avatars = None
        else:
            self.avatars = nn.ModuleList(avatar)
            self.avatar = self.avatars[0]
        
        self.background = background
        self.pure_colors = PureColorBackground()
        
        # Renderer
        self.renderer = GaussianRenderer(
            sh_levels=cfg.render.sh_levels,
            bg_color=cfg.render.bg_color,
        )
        self.use_zero_scales = cfg.render.use_zero_scales
        self.use_constant_colors = cfg.render.use_constant_colors
        self.use_constant_opacities = cfg.render.use_constant_opacities
        if self.use_constant_colors is not None:
            self.constant_colors = torch.tensor([self.use_constant_colors], device=self.device)
            self.use_constant_colors = True
        else:
            self.constant_colors = None
            self.use_constant_colors = False
        if self.use_constant_opacities is not None:
            self.constant_opacities = torch.tensor([self.use_constant_opacities], device=self.device)
            self.use_constant_opacities = True
        else:
            self.constant_opacities = None
            self.use_constant_opacities = False

        self.use_fixed_n_gaussians = cfg.render.use_fixed_n_gaussians
        if self.use_fixed_n_gaussians is not None:
            self.fixed_n_gaussians = int(self.use_fixed_n_gaussians)
            self.use_fixed_n_gaussians = True
        else:
            self.fixed_n_gaussians = None
            self.use_fixed_n_gaussians = False
        self.avatar_transl = torch.tensor(eval(cfg.render.avatar_transl), device=self.device) if cfg.render.avatar_transl is not None else None
        self.avatar_scale = torch.tensor(eval(cfg.render.avatar_scale), device=self.device) if cfg.render.avatar_scale is not None else None

    def avatar_forward(
        self,
        smpl_observed_inputs: Optional[dict] = None,
        avatar: Optional[Avatar] = None,
        avatar_index: Optional[int] = None,
    ) -> GaussianOutput:
        if avatar is None:
            avatar = self.avatar
        if smpl_observed_inputs is None:
            gaussians = avatar.forward()
        else:
            gaussians = avatar.animate(smpl_observed_inputs=smpl_observed_inputs)
        
        if self.avatar_scale is not None:
            avatar_scale = self.avatar_scale
            if avatar_scale.ndim == 1:
                avatar_scale = avatar_scale[avatar_index]
            gaussians.positions = gaussians.positions * avatar_scale.unsqueeze(0)
            gaussians.scales = gaussians.scales * avatar_scale.unsqueeze(0)

        if self.avatar_transl is not None:
            avatar_transl = self.avatar_transl
            if avatar_transl.ndim == 2:
                avatar_transl = avatar_transl[avatar_index]
            gaussians.positions = gaussians.positions + avatar_transl.unsqueeze(0)

        return gaussians

    def forward(
        self,
        data: dict,
        smpl_observed_inputs: Optional[dict] = None,
        use_densifier: bool = True,
        bg_mode: Optional[str] = None,
        **kwargs,
    ):
        if self.avatars is None:
            gaussians = self.avatar_forward(smpl_observed_inputs=smpl_observed_inputs)
        else:
            gaussians_list = []
            batch_size = smpl_observed_inputs['body_pose'].size(0)
            assert batch_size <= len(self.avatars), \
                f'Assert num_smplx_inputs: {batch_size} <= num_avatars: {len(self.avatars)}'
            for i, avatar in enumerate(self.avatars):
                smpl_inputs = {}
                for k, v in smpl_observed_inputs.items():
                    smpl_inputs[k] = v[i:i+1, ...]
                gaussians = self.avatar_forward(
                    smpl_observed_inputs=smpl_inputs,
                    avatar=avatar,
                    avatar_index=i,
                )
                gaussians_list.append(gaussians)
            gaussians = merge_gaussians(*gaussians_list)
        
        if type(self.background) is GaussianModel:
            bg_gaussians = self.background.forward()
            bg_gaussians.colors = self.renderer.compute_colors(
                sh_features=bg_gaussians.sh_features,
                positions=bg_gaussians.positions,
                camera_positions=data['c2w'][:, :3, 3],
                sh_levels=1,
            )
            bg_gaussians.sh_features = None
            gaussians = merge_gaussians(gaussians, bg_gaussians)
        
        if self.use_zero_scales:
            # gaussians.scales = torch.zeros_like(gaussians.scales)
            gaussians.scales = gaussians.scales * 0.1

        if self.use_constant_colors:
            gaussians.colors = self.constant_colors.expand(gaussians.colors.size(0), -1)
        
        if self.use_constant_opacities:
            gaussians.opacities = self.constant_opacities.expand(gaussians.opacities.size(0), -1)
        
        if self.use_fixed_n_gaussians:
            gaussians = downsample_gaussians(gaussians, self.fixed_n_gaussians)

        outputs = self.renderer.render(
            data=data,
            gaussians=gaussians,
            return_2d_radii=use_densifier,
        )

        if bg_mode in self.pure_colors:
            outputs['image_bg'] = self.pure_colors.get_background_like(bg_mode, outputs['image'])
            outputs['image_fg'] = outputs['image']
            outputs['image'] = outputs['image'] + outputs['image_bg']  * (1 - outputs['alpha'])
        elif type(self.background) is VideoBackground:
            outputs['image_bg'] = self.background.get_background_like(data['frame_index'], outputs['image'])
            outputs['image_fg'] = outputs['image']
            outputs['image'] = outputs['image'] + outputs['image_bg']  * (1 - outputs['alpha'])
        elif type(self.background) is MLPBackground:
            outputs['image_bg'] = self.background(data)
            outputs['image_fg'] = outputs['image']
            outputs['image'] = outputs['image'] + outputs['image_bg']  * (1 - outputs['alpha'])
        else:
            outputs['image_fg'] = outputs['image']

        return outputs
    
    def densify(self, densifiers: dict, render_outputs: dict, spatial_scale:float, train_step:int):
        
        if hasattr(self.avatar, 'densification_mask'):
            densification_mask = self.avatar.densification_mask
            viewspace_points = render_outputs['viewspace_points'][densification_mask]
            viewspace_points.grad = render_outputs['viewspace_points'].grad[densification_mask]
            radii = render_outputs['radii'][densification_mask]
        else:
            viewspace_points = render_outputs['viewspace_points']
            radii = render_outputs['radii']
        
        densifiers['avatar'](
            viewspace_points=viewspace_points,
            radii=radii,
            spatial_extent=spatial_scale,
            train_step=train_step,
        )

    def organize_state_dict(self, state_dict: Mapping[str, Any]):
        state_dict_by_modules = defaultdict(dict)
        for k, v in state_dict.items():
            module_name = k.split('.')[0]
            param_name = k.replace(f'{module_name}.', '', 1)
            state_dict_by_modules[module_name][param_name] = v
        return state_dict_by_modules

    def fix_state_dict(self, state_dict: Mapping[str, Any], reset_nerf_mlp: bool = False):
        if reset_nerf_mlp:
            pop_keys = []
            for k in state_dict.keys():
                if k.startswith('avatar.nerf_opacity_and_color_net.'):
                    print(k)
                    pop_keys.append(k)
            for k in pop_keys:
                state_dict.pop(k)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict:bool=True):
        state_dict = self.fix_state_dict(state_dict)
        state_dict_by_modules = self.organize_state_dict(state_dict)
        self.avatar.reset_by_state_dict(state_dict_by_modules['avatar'])
        return super().load_state_dict(state_dict, strict=strict)

    def load_extra_avatar_from_state_dict(self, state_dict: Mapping[str, Any], strict:bool=True):
        state_dict = self.fix_state_dict(state_dict)
        state_dict_by_modules = self.organize_state_dict(state_dict)
        new_avatar = deepcopy(self.avatar)
        new_avatar.reset_by_state_dict(state_dict_by_modules['avatar'])
        new_avatar.load_state_dict(state_dict_by_modules['avatar'], strict=strict)
        if self.avatars is not None:
            self.avatars.append(new_avatar)
        else:
            self.avatars = nn.ModuleList((self.avatar, new_avatar))


def build_scene(cfg: TrainConfig, avatar: Avatar | Iterable[Avatar]):

    if cfg.render.use_mlp_background:
        background = MLPBackground()
    
    elif cfg.render.use_video_background:
        background = VideoBackground(cfg.render.use_video_background)

    elif cfg.render.use_gs_background:
        background = GaussianModel()
        background.load_ply_and_initialize(cfg.render.use_gs_background)
    
    else:
        background = None

    return Scene(
        cfg=cfg,
        avatar=avatar,
        background=background,
    ).to(torch.device(cfg.device))
