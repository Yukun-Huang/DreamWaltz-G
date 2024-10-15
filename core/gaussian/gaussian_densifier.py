import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from loguru import logger
from typing import Union

from core.gaussian.gaussian_optimizer import GaussianModel, GaussianOptimizer
from configs import TrainConfig


class DensificationParams:
    def __init__(
        self,
        max_iteration: int,
        densify_from_iter: int = None,
        densify_until_iter: int = None,
        densification_interval: int = None,
        opacity_reset_interval: int = None,
        densify_grad_threshold: float = 0.0002,
        prune_opacity_threshold: float = 0.005,
        densify_screen_size_threshold: float = 20.0,
        densification_percent_distinction: float = 0.01,
        disable_densify_clone: bool = False,
        disable_densify_split: bool = False,
        disable_prune: bool = False,
        disable_reset: bool = False,
        enable_grad_prune: bool = False,
    ):
            
        if densify_from_iter is None:
            densify_from_iter = int(max_iteration * 500 / 15000)
        
        if densify_until_iter is None:
            densify_until_iter = int(max_iteration * 7000 / 15000)

        if densification_interval is None:
            densification_interval = int(max_iteration * 100 / 15000)
    
        if opacity_reset_interval is None:
            opacity_reset_interval = int(max_iteration * 3000 / 15000)

        self.max_iteration = max_iteration

        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter

        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval

        self.densify_grad_threshold = densify_grad_threshold
        self.prune_opacity_threshold = prune_opacity_threshold

        self.densify_screen_size_threshold = densify_screen_size_threshold
        self.densification_percent_distinction = densification_percent_distinction

        self.disable_densify_clone = disable_densify_clone
        self.disable_densify_split = disable_densify_split
        self.disable_prune = disable_prune
        self.disable_reset = disable_reset

        self.enable_grad_prune = enable_grad_prune

    def __str__(self):
        return f"""DensificationParams(
            max_iteration={self.max_iteration},
            densify_from_iter={self.densify_from_iter},
            densify_until_iter={self.densify_until_iter},
            densification_interval={self.densification_interval},
            opacity_reset_interval={self.opacity_reset_interval},
            densify_grad_threshold={self.densify_grad_threshold},
            prune_opacity_threshold={self.prune_opacity_threshold},
            densify_screen_size_threshold={self.densify_screen_size_threshold},
            densification_percent_distinction={self.densification_percent_distinction},
            disable_densify_clone={self.disable_densify_clone},
            disable_densify_split={self.disable_densify_split},
            disable_prune={self.disable_prune},
            disable_reset={self.disable_reset},
            enable_grad_prune={self.enable_grad_prune},
        )"""


class GaussianDensifier:
    def __init__(
        self,
        model: GaussianModel,
        params: DensificationParams,
        optimizer: GaussianOptimizer,
    ) -> None:
        pass
        self.model = model
        self.device = model.device
        self.params = params
        self.optimizer = optimizer.optimizer
        
        self.points_gradient_accum = torch.zeros((self.model._n_points, 1), device=self.device)
        self.denom = torch.zeros((self.model._n_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.model._n_points,), device=self.device)
        
        self.spatial_extent = None

        self.densify_from_iter = params.densify_from_iter
        self.densify_until_iter = params.densify_until_iter
        self.densification_interval = params.densification_interval
        self.opacity_reset_interval = params.opacity_reset_interval

        self.max_grad = params.densify_grad_threshold
        self.max_screen_size = params.densify_screen_size_threshold
        self.percent_dense = params.densification_percent_distinction
        self.min_opacity = params.prune_opacity_threshold

        self.disable_densify_clone = params.disable_densify_clone
        self.disable_densify_split = params.disable_densify_split
        self.disable_prune = params.disable_prune
        self.disable_reset = params.disable_reset

        self.enable_grad_prune = params.enable_grad_prune

        self.params_to_densify = optimizer.param_names
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in self.params_to_densify:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name in self.params_to_densify:
                assert len(group["params"]) == 1

                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                dim_to_cat = 0
                
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=dim_to_cat)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=dim_to_cat)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=dim_to_cat).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=dim_to_cat).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)

                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def update_model(self, optimizable_tensors):

        for k, v in optimizable_tensors.items():
            setattr(self.model, f'_{k}', v)
        
        self.model._n_points = len(self.model._positions)

    def densification_postfix(self, tensors_dict):

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)
        
        self.update_model(optimizable_tensors=optimizable_tensors)

        self.points_gradient_accum = torch.zeros((self.model._n_points, 1), device=self.device)
        self.denom = torch.zeros((self.model._n_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.model._n_points), device=self.device)
    
    def update_densification_stats(self, viewspace_point_tensor, radii, visibility_filter):
        # Updates maximum observed 2D radii of all gaussians
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
        
        # Accumulate gradient magnitudes of all points
        self.points_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)
        
        # Counts number of updates for each point
        self.denom[visibility_filter] += 1

    def get_prune_mask(self, extent: float, grads: torch.Tensor = None):
        min_opacity = self.min_opacity
        max_screen_size = self.max_screen_size
        
        with torch.no_grad():
            scales = self.model.get_scales()
            opacities = self.model.get_opacities()
        
        prune_mask = (opacities < min_opacity).squeeze()
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = scales.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if grads is not None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= self.max_grad, True, False)
            prune_mask = torch.logical_or(prune_mask, selected_pts_mask)
        
        return prune_mask

    def densify_and_clone(self, grads: torch.Tensor, extent: float):
        with torch.no_grad():
            scales = self.model.get_scales()

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= self.max_grad, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values <= self.percent_dense * extent,
        )
        
        tensors_dict = {}
        for param_name in self.params_to_densify:
            tensors_dict[param_name] = getattr(self.model, f'_{param_name}')[selected_pts_mask]
        
        self.densification_postfix(tensors_dict)

        if hasattr(self.model, 'vertex_indices'):
            new_vertex_indices = self.model.vertex_indices[selected_pts_mask.detach().cpu()]
            self.model.vertex_indices = torch.cat((self.model.vertex_indices, new_vertex_indices), dim=0)
        
        if hasattr(self.model, '_lbs_weights'):
            assert self.model._lbs_weights.requires_grad is False
            old_lbs_weights = self.model._lbs_weights.data
            new_lbs_weights = self.model._lbs_weights[selected_pts_mask]
            self.model._lbs_weights.data = torch.cat((old_lbs_weights, new_lbs_weights), dim=0)
    
    def densify_and_split(self, grads: torch.Tensor, extent: float, N: int = 2):
        # Get current gaussians
        with torch.no_grad():
            quaternions = self.model.get_quaternions()
            scales = self.model.get_scales()
        
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(self.model._n_points, device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= self.max_grad, True, False)  # [N=10391,]
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values > self.percent_dense * extent,  # [8029, 3]
        )

        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = quaternion_to_matrix(quaternions[selected_pts_mask]).repeat(N, 1, 1)
        
        tensors_dict = {}
        for param_name in self.params_to_densify:
            params = getattr(self.model, f'_{param_name}')[selected_pts_mask]
            tensors_dict[param_name] = params.repeat(N, *([1]*(params.ndim-1)))

        tensors_dict['positions'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        tensors_dict['scales'] = self.model.scale_inverse_activation(self.model.scale_activation(tensors_dict['scales']) / (0.8*N))
        
        self.densification_postfix(tensors_dict)

        if hasattr(self.model, 'vertex_indices'):
            new_vertex_indices = (self.model.vertex_indices[selected_pts_mask.detach().cpu()]).repeat(N, )
            self.model.vertex_indices = torch.cat((self.model.vertex_indices, new_vertex_indices), dim=0)
        
        if hasattr(self.model, '_lbs_weights'):
            assert self.model._lbs_weights.requires_grad is False
            old_lbs_weights = self.model._lbs_weights.data
            new_lbs_weights = (self.model._lbs_weights[selected_pts_mask]).repeat(N, 1)
            self.model._lbs_weights.data = torch.cat((old_lbs_weights, new_lbs_weights), dim=0)
    
        prune_mask = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool),
        ))
        self.prune(prune_mask)
    
    def prune(self, prune_mask: torch.Tensor):

        valid_mask = ~prune_mask

        optimizable_tensors = self._prune_optimizer(valid_mask)
        
        self.update_model(optimizable_tensors=optimizable_tensors)
        
        self.points_gradient_accum = self.points_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]

        valid_mask = valid_mask.detach().cpu()

        if hasattr(self.model, 'vertex_indices'):
            self.model.vertex_indices = self.model.vertex_indices[valid_mask]
        
        if hasattr(self.model, '_lbs_weights'):
            if self.model._lbs_weights.shape[0] != valid_mask.shape[0]:
                import pdb;pdb.set_trace()
            assert self.model._lbs_weights.requires_grad is False
            self.model._lbs_weights.data = self.model._lbs_weights.data[valid_mask]

    def reset_opacity(self, value: float = 0.01):
        with torch.no_grad():
            opacities = self.model.get_opacities()
            opacities = self.model.opacity_inverse_activation(
                torch.min(opacities, torch.ones_like(opacities) * value)
            )
        assert self.model._opacities is not None
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities, "opacities")
        self.model._opacities = optimizable_tensors["opacities"]

    @torch.no_grad()
    def __call__(
        self,
        viewspace_points: torch.Tensor,
        radii: torch.Tensor,
        spatial_extent: float,
        train_step: int,
    ):
        if train_step >= self.densify_until_iter:
            return

        self.update_densification_stats(viewspace_points, radii, visibility_filter=radii > 0)

        if train_step > self.densify_from_iter and train_step % self.densification_interval == 0:

            if spatial_extent is None:
                spatial_extent = self.spatial_extent

            grads = self.points_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            
            n_points = self.model._n_points

            if not self.enable_grad_prune:
                if not self.disable_densify_clone:
                    self.densify_and_clone(grads=grads, extent=spatial_extent)
                if not self.disable_densify_split:
                    self.densify_and_split(grads=grads, extent=spatial_extent)

            new_points = self.model._n_points - n_points

            if not self.disable_prune:
                if self.enable_grad_prune:
                    prune_mask = self.get_prune_mask(extent=spatial_extent, grads=grads)
                    grad_prune_iters = (self.densify_until_iter - self.densify_from_iter) / 3
                    if train_step > (self.densify_from_iter + grad_prune_iters):
                        self.enable_grad_prune = False
                else:
                    prune_mask = self.get_prune_mask(extent=spatial_extent)
                self.prune(prune_mask)

            pruned_points = n_points + new_points - self.model._n_points

            torch.cuda.empty_cache()

            logger.info("Gaussians densified and pruned. Number of gaussians: "
                       f"{n_points} + {new_points} - {pruned_points} = {self.model._n_points}")
        
        if not self.disable_reset and train_step % self.opacity_reset_interval == 0:
            self.reset_opacity()
            logger.info("Opacity reset.")


def build_densifier(model: GaussianModel, optimizer: GaussianOptimizer, cfg: TrainConfig):

    densify_params = DensificationParams(
        max_iteration=cfg.optim.iters,
        densify_from_iter=cfg.render.densify_from_iter,
        densify_until_iter=cfg.render.densify_until_iter,
        densify_grad_threshold=cfg.render.densify_grad_threshold,
        disable_densify_clone=cfg.render.densify_disable_clone,
        disable_densify_split=cfg.render.densify_disable_split,
        disable_prune=cfg.render.densify_disable_prune,
        disable_reset=cfg.render.densify_disable_reset,
        enable_grad_prune=cfg.render.enable_grad_prune,
    )

    logger.info(f'Initialize Densifier of Gaussian Model: {model.__class__.__name__}')
    logger.info(f'Densifier Hyperparameters: {densify_params}')

    return GaussianDensifier(
        model=model,
        optimizer=optimizer,
        params=densify_params,
    )
