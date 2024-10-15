import torch
import numpy as np
from loguru import logger
from typing import Optional

from .gaussian_model import GaussianModel
from configs import TrainConfig
from core.optim.optim_utils import get_expon_lr_func


class OptimizationParams:
    def __init__(self, 
                 iterations:int,
                 position_lr_init:float,
                 position_lr_final:float,
                 position_lr_delay_mult:float,
                 position_lr_max_steps:int,
                 feature_lr:float,
                 opacity_lr:float,
                 scaling_lr:float,
                 rotation_lr:float,
        ):
        
        # Basic Gaussian Splatting
        self.iterations = iterations
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr

    def __str__(self):
        return f"""OptimizationParams(
            iterations={self.iterations},
            position_lr_init={self.position_lr_init},
            position_lr_final={self.position_lr_final},
            position_lr_delay_mult={self.position_lr_delay_mult},
            position_lr_max_steps={self.position_lr_max_steps},
            feature_lr={self.feature_lr},
            opacity_lr={self.opacity_lr},
            scaling_lr={self.scaling_lr},
            rotation_lr={self.rotation_lr},
            )"""


class GaussianOptimizer:

    def __init__(
        self,
        model: GaussianModel,
        params: Optional[OptimizationParams] = None,
    ) -> None:

        self.current_iteration = 0
        self.num_iterations = params.iterations

        self.param_names = []
        
        if params is None:
            params = OptimizationParams()
        
        l = []

        if model._positions is not None and model._positions.requires_grad:
            l += [{'params': [model._positions], 'lr': params.position_lr_init, 'name': "positions"}]
            self.param_names.append('positions')

        if model._sh_features_dc is not None and model._sh_features_dc.requires_grad:
            l += [{'params': [model._sh_features_dc], 'lr': params.feature_lr, 'name': "sh_features_dc"}]
            self.param_names.append("sh_features_dc")

        if model._sh_features_rest is not None and model._sh_features_rest.requires_grad:
            l += [{'params': [model._sh_features_rest], 'lr': params.feature_lr / 20.0, 'name': "sh_features_rest"}]
            self.param_names.append("sh_features_rest")

        if model._opacities is not None and model._opacities.requires_grad:
            l += [{'params': [model._opacities], 'lr': params.opacity_lr, 'name': "opacities"}]
            self.param_names.append("opacities")

        if model._scales is not None and model._scales.requires_grad:
            self.default_scaling_lr = params.scaling_lr
            l += [{'params': [model._scales], 'lr': params.scaling_lr, 'name': "scales"}]
            self.param_names.append("scales")

        if model._quaternions is not None and model._quaternions.requires_grad:
            l += [{'params': [model._quaternions], 'lr': params.rotation_lr, 'name': "quaternions"}]
            self.param_names.append("quaternions")
        
        if len(l) > 0:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = None
        
        self.position_sheduler_func = get_expon_lr_func(
            lr_init=params.position_lr_init, 
            lr_final=params.position_lr_final, 
            lr_delay_mult=params.position_lr_delay_mult, 
            max_steps=params.position_lr_max_steps
        )

        # self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
        #                                             lr_final=training_args.rotation_lr_final,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.iterations)

        # self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr,
        #                                             lr_final=training_args.scaling_lr_final,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.iterations)

        # self.feature_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr,
        #                                             lr_final=training_args.feature_lr_final,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.iterations)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self.optimizer.step()
        self.current_iteration += 1
        
    def zero_grad(self, set_to_none:bool=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
    def update_learning_rate(self, spatial_scale: float, iteration: Optional[int] = None):
        if iteration is None:
            iteration = self.current_iteration
        lr = 0.
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == "positions":
                lr = self.position_sheduler_func(iteration)
                param_group['lr'] = lr * spatial_scale
            elif param_group['name'] == "scales":
                lr = self.default_scaling_lr
                param_group['lr'] = lr * spatial_scale
        return lr
    
    def add_param_group(self, new_param_group):
        self.optimizer.add_param_group(new_param_group)

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def build_optimizer(model, cfg: TrainConfig, optim_params: Optional[OptimizationParams] = None):
    
    if optim_params is None:
        iterations = cfg.optim.iters
        optim_params = OptimizationParams(
            iterations=iterations,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=iterations * 2,
            feature_lr=0.0125,
            opacity_lr=0.01,
            scaling_lr=0.005,
            rotation_lr=0.001,
        )
    logger.info(f'Initialize Optimizer of Gaussian Model: {model.__class__.__name__}')
    logger.info(f'Optimizer Hyperparameters: {optim_params}')

    gaussian_optimizer = GaussianOptimizer(model=model, params=optim_params)
    if gaussian_optimizer.optimizer is None:
        logger.warning('No 3D gaussian parameters to optimize')
        return None
    else:
        return gaussian_optimizer
