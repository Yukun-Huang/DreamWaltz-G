import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion, quaternion_multiply
from typing import Tuple, Mapping, Any, Optional, List, Dict
from collections import defaultdict
from loguru import logger

from configs import TrainConfig
from core.gaussian.gaussian_utils import GaussianOutput, merge_gaussians
from core.gaussian.gaussian_model import GaussianModel
from core.gaussian.gaussian_optimizer import OptimizationParams, GaussianOptimizer, build_optimizer
from core.gaussian.gaussian_densifier import build_densifier
from core.gaussian.spherical_harmonics import RGB2SH
from core.deformation.deform_model import DeformNetwork
from core.human.smpl_prompt import SMPLPrompt
from core.human.inverse_lbs import LinearBlendSkinning, GeneralLinearBlendSkinning, RigidTransform
from utils.point_cloud import BasicPointCloud
from utils.mesh import compute_normal
from core.nerf.nerf_model import NeRFNetwork, MLP, build_NeRFNetwork


def knn_points(query_points, reference_points, K=3, device=None):
    from pytorch3d.ops import knn_points as _knn_points
    from pytorch3d.ops.knn import _KNN
    if device is None:
        device = query_points.device
    res = _knn_points(query_points.cuda(), reference_points.cuda(), K=K)
    return _KNN(
        dists=res.dists.to(device),
        idx=res.idx.to(device),
        knn=None,
    )


####################################################################################
class _Avatar(GaussianModel):
    def __init__(
        self,
        cfg: TrainConfig,
        canonical_vertices: torch.Tensor,
        canonical_triangles: torch.Tensor,
    ) -> None:
        """
        Args:
            canonical_vertices: torch.Tensor, [V, 3]
            canonical_triangles: torch.Tensor, [F, 3]
        """
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.sh_levels = cfg.render.sh_levels
        
        # Register vertices and triangles of SMPLX
        assert canonical_vertices.ndim == 2 and canonical_triangles.ndim == 2
        self.n_vertices = len(canonical_vertices)
        self.n_triangles = len(canonical_triangles)
        self.canonical_vertices = canonical_vertices.detach().cpu()
        self.canonical_triangles = canonical_triangles.detach().cpu()
        
        # Initialize regularizations
        self.regularizations = {}

    def initialize_positions(
        self,
        cfg: TrainConfig,
        vertices: Optional[torch.Tensor] = None,
        triangles: Optional[torch.Tensor] = None,
        point_cloud: Optional[BasicPointCloud] = None,
    ) -> torch.Tensor:
        """
        Initializes the positions of the avatar based on the given configuration and input data.

        Args:
            cfg (TrainConfig): The training configuration.
            vertices (Optional[torch.Tensor]): The vertices of the avatar mesh.
            triangles (Optional[torch.Tensor]): The triangles of the avatar mesh.
            point_cloud (Optional[BasicPointCloud]): The point cloud representing the avatar.

        Returns:
            torch.Tensor: The initialized positions of the avatar.

        Raises:
            NotImplementedError: If the 'gaussian_point_init' is set to 'mesh_triangle'.

        """
        gaussian_point_init = cfg.render.gaussian_point_init
        
        if point_cloud is not None:
            # Initialize positions from the given point cloud
            positions = torch.tensor(point_cloud.points).float()

        elif gaussian_point_init == 'mesh_surface':
            # Initialize positions by sampling points on the surface of the mesh
            import trimesh
            mesh = trimesh.Trimesh(vertices.numpy(), triangles.numpy())
            positions, _ = trimesh.sample.sample_surface(mesh, cfg.render.n_gaussians)
            positions = torch.tensor(positions).float()

        elif gaussian_point_init == 'mesh_vertex':
            # Initialize positions by selecting points from the vertices of the mesh
            n_vertices = len(vertices)
            vertex_indices = [[i] * cfg.render.n_gaussians_per_vertex for i in range(n_vertices)]
            vertex_indices = torch.tensor(vertex_indices).flatten()
            positions = vertices[vertex_indices]
        
        elif gaussian_point_init == 'mesh_triangle':
            raise NotImplementedError("Initialization method 'mesh_triangle' is not implemented.")
        
        else:
            assert 0, gaussian_point_init
        
        return positions

    def initialize_colors(
        self,
        cfg: TrainConfig,
        n_points: int,
        point_cloud: Optional[BasicPointCloud] = None,
        vertices: Optional[torch.Tensor] = None,
        triangles: Optional[torch.Tensor] = None,
        vertex_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initializes the colors for the avatar.

        Args:
            cfg (TrainConfig): The training configuration.
            n_points (int): The number of points.
            point_cloud (Optional[BasicPointCloud]): The point cloud.
            vertices (Optional[torch.Tensor]): The vertices tensor.
            triangles (Optional[torch.Tensor]): The triangles tensor.
            vertex_indices (Optional[torch.Tensor]): The vertex indices tensor.

        Returns:
            torch.Tensor: The initialized colors tensor.
        """
        gaussian_color_init = cfg.render.gaussian_color_init
        
        if point_cloud is not None:
            colors = torch.tensor(point_cloud.colors).float()
        
        elif gaussian_color_init == 'rand':
            colors = torch.rand(n_points, 3)
        
        elif gaussian_color_init == 'constant':
            colors = torch.ones(n_points, 3) * 0.5
        
        elif gaussian_color_init == 'ones':
            colors = torch.ones(n_points, 3)
        
        elif gaussian_color_init == 'normal':
            from utils.mesh import compute_normal
            vn, _ = compute_normal(vertices, triangles)
            colors = ((vn + 1) / 2.).mean(dim=1, keepdim=True).repeat(1, 3)[vertex_indices]
        
        else:
            assert 0, gaussian_color_init
        
        return colors

    def initialize_radiuses(
        self,
        vertices: torch.Tensor,
        vertex_indices: torch.Tensor,
        use_sqrt=True,
        use_mean=False,
        K=3,
        init_radius_norm=1.,
    ) -> torch.Tensor:
        """Function to initialize the radiuses, borrowed from SuGaR.

        Args:
            vertices: Tensor with shape (V, 3)
            vertex_indices: Tensor with shape (N,)
        
        Returns:
            radiuses: Tensor with shape (N, 1)
        """
        device = vertices.device

        # Initialize learnable radiuses
        knn_dists = knn_points(vertices[None], vertices[None], K=K+1).dists[0, :, 1:]  # [V, K-1]
        if use_sqrt:
            knn_dists = torch.sqrt(knn_dists)
        
        if use_mean:
            radiuses = knn_dists.mean(-1, keepdim=True)  # [V, 1]
        else:
            radiuses = knn_dists.min(-1, keepdim=True)[0]  # [V, 1]

        radiuses = radiuses.clamp_min(1e-7) * init_radius_norm
        return radiuses[vertex_indices, :].to(device)  # [N, 1]

    def initialize_scales(
        self,
        cfg: TrainConfig,
        n_points: int,
        vertices: torch.Tensor,
        vertex_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initializes the scales for the avatar.

        Args:
            cfg (TrainConfig): The configuration object for training.
            n_points (int): The number of points in the avatar.
            vertices (torch.Tensor): The tensor containing the vertices of the avatar.
            vertex_indices (torch.Tensor): The tensor containing the indices of the vertices.

        Returns:
            torch.Tensor: The tensor containing the initialized scales for the avatar.
        """
        gaussian_scale_init = cfg.render.gaussian_scale_init
        if gaussian_scale_init == 'radius':
            radiuses = self.initialize_radiuses(vertices, vertex_indices)
            scales = self.scale_inverse_activation(radiuses.expand(-1, 3) * cfg.render.init_scale_radius_rate)
        else:
            scales = self.scale_inverse_activation(torch.ones(n_points, 3) * cfg.render.init_scale)
        return scales

    def initialize_vertex_indices(self, positions: torch.Tensor, vertices: torch.Tensor):
        return knn_points(
            positions[None].data.detach().cpu(),
            vertices[None].detach().cpu(),
            K=1,
        ).idx[0, :, 0]

    def get_densifier(self, cfg:TrainConfig, optimizer: GaussianOptimizer):
        return build_densifier(
            model=self,
            optimizer=optimizer,
            cfg=cfg,
        )

    def get_optimizer(self, cfg:TrainConfig) -> dict:
        optimizers = {}
        iterations = cfg.optim.iters
        optim_params = OptimizationParams(
            iterations=iterations,
            position_lr_init=cfg.render.position_lr_init,
            position_lr_final=cfg.render.position_lr_final,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=iterations * 2,
            feature_lr=cfg.render.feature_lr,
            opacity_lr=cfg.render.opacity_lr,
            scaling_lr=cfg.render.scaling_lr,
            rotation_lr=cfg.render.rotation_lr,
        )
        optimizers['avatar'] = build_optimizer(model=self, cfg=cfg, optim_params=optim_params)
        return optimizers

    def get_vertex_indices(self) -> torch.Tensor:
        return self.vertex_indices


class _AnimatableAvatar(_Avatar):
    def __init__(
        self,
        cfg: TrainConfig,
        lbs_model: Optional[LinearBlendSkinning] = None,
        deform_model: Optional[DeformNetwork] = None,
        smpl_canonical_inputs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(cfg=cfg, **kwargs)
        self.lbs_model = lbs_model
        self.deform_model = deform_model
        self.deform_rotation_mode = cfg.render.deform_rotation_mode
        self.deform_with_shape = cfg.render.deform_with_shape

        self.smpl_canonical_inputs = smpl_canonical_inputs
    
    def get_vertex_indices(self) -> torch.Tensor:
        raise NotImplementedError

    def lbs_transform(self, gaussians: GaussianOutput, smpl_observed_inputs: dict) -> GaussianOutput:
        # Transform the gaussians from smpl-canonical to canonical space
        canonical_transform: RigidTransform = self.lbs_model.forward(
            **self.smpl_canonical_inputs,
            deform_with_shape=self.deform_with_shape,
        )
        # Transform the gaussians from smpl-canonical to observed space
        observed_transform: RigidTransform = self.lbs_model.forward(
            **smpl_observed_inputs,
            deform_with_shape=self.deform_with_shape,
        )
        # Compute the transform from canonical to observed space
        cnl2obs_transform: RigidTransform = canonical_transform.inverse().compose(observed_transform).squeeze(0)
        
        indices = self.get_vertex_indices()

        gaussians.positions = cnl2obs_transform.transform_points(
            pts=gaussians.positions,
            indices=indices,
        )
        gaussians.quaternions = cnl2obs_transform.transform_quaternions(
            quaternions=gaussians.quaternions,
            indices=indices,
            rotation_mode=self.deform_rotation_mode,
        )
        return gaussians

    def animate(self, smpl_observed_inputs: dict, **kwargs) -> GaussianOutput:
        """
        Args:
            body_pose: torch.Tensor, shape = (N, V*3) or (N, V, 3, 3)
            global_orient: torch.Tensor, shape = (N, 3) or (N, 3, 3)
            betas: torch.Tensor, shape = (N, 10)
            transl: torch.Tensor, shape = (N, 3)
        """
        gaussians = self.forward()
        if smpl_observed_inputs is not None:
            if self.lbs_model is not None:
                gaussians = self.lbs_transform(gaussians, smpl_observed_inputs)
            if self.deform_model is not None:
                raise NotImplementedError
        return gaussians

    def get_optimizer(self, cfg: TrainConfig) -> dict:
        optimizers = super().get_optimizer(cfg=cfg)
        if self.lbs_model is not None:
            pass
            # lbs_optimizer = self.lbs_model.get_optimizer(lr=cfg.render.lbs_lr)
            # if lbs_optimizer is not None:
            #     optimizers['lbs'] = lbs_optimizer
        if self.deform_model is not None:
            optimizers['deform'] = self.deform_model.get_optimizer(cfg=cfg)
        return optimizers


####################################################################################
class VanillaAvatar(_AnimatableAvatar):
    def __init__(
        self,
        cfg: TrainConfig,
        point_cloud: BasicPointCloud = None,
        learn_positions: bool = True,
        learn_colors: bool = True,
        learn_opacities: bool = True,
        learn_scales: bool = True,
        learn_quaternions: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            cfg=cfg,
            **kwargs,
        )

        # Initialize learnable parameter flags
        self.learn_positions = learn_positions
        self.learn_colors = learn_colors
        self.learn_opacities = learn_opacities
        self.learn_scales = learn_scales
        self.learn_quaternions = learn_quaternions

        # Initialize positions
        positions = self.initialize_positions(cfg=cfg,
            vertices=self.canonical_vertices,
            triangles=self.canonical_triangles,
            point_cloud=point_cloud,
        )
        self._positions = nn.Parameter(positions, requires_grad=learn_positions)
        self._n_points = len(positions)

        # Initialize vertex indices and densification mask
        self.vertex_indices = self.initialize_vertex_indices(positions, self.canonical_vertices)

        # Initialize SH features
        colors = self.initialize_colors(cfg=cfg, n_points=self._n_points, point_cloud=point_cloud)
        self._sh_features_dc = nn.Parameter(RGB2SH(colors).unsqueeze(dim=1), requires_grad=learn_colors)
        self._sh_features_rest = nn.Parameter(torch.zeros(self._n_points, self.sh_levels**2 - 1, 3), requires_grad=learn_colors)

        # Initialize opacities
        opacities = self.opacity_inverse_activation(cfg.render.init_opacity * torch.ones((self._n_points, 1)))
        self._opacities = nn.Parameter(opacities, requires_grad=learn_opacities)

        # Initialize scales
        scales = self.initialize_scales(
            cfg=cfg,
            n_points=self._n_points,
            vertices=self.canonical_vertices,
            vertex_indices=self.vertex_indices,
        )
        self._scales = nn.Parameter(scales, requires_grad=learn_scales)

        # Initialize quaternions
        quaternions = matrix_to_quaternion(torch.eye(3)[None].repeat(self._n_points, 1, 1))  # [N, 4]
        self._quaternions = nn.Parameter(quaternions, requires_grad=learn_quaternions)

    def reset_by_state_dict(self, state_dict: Mapping[str, Any]):
        super().reset_by_state_dict(state_dict=state_dict)
        self.vertex_indices = self.initialize_vertex_indices(state_dict['_positions'], self.canonical_vertices)

    def forward(self) -> GaussianOutput:
        return GaussianOutput(
            positions=self.get_positions(),
            sh_features=self.get_sh_features(),
            opacities=self.get_opacities(),
            quaternions=self.get_quaternions(),
            scales=self.get_scales(),
        )

    @property
    def densification_mask(self):
        return torch.ones(self._n_points, dtype=torch.bool)


####################################################################################
class HashAvatar(_AnimatableAvatar):
    def __init__(
        self,
        cfg: TrainConfig,
        nerf: NeRFNetwork,
        point_cloud: BasicPointCloud = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Avatar class.

        Args:
            cfg (TrainConfig): The configuration object.
            nerf (NeRFNetwork, optional): The NeRFNetwork object. Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(cfg=cfg, **kwargs)

        # Initialize positions
        positions = self.initialize_positions(cfg=cfg,
            vertices=self.canonical_vertices,
            triangles=self.canonical_triangles,
            point_cloud=point_cloud,
        )
        self._positions = nn.Parameter(positions, requires_grad=True)
        self._n_points = len(positions)

        # Initialize vertex indices and densification mask
        self.vertex_indices = self.initialize_vertex_indices(positions, self.canonical_vertices)

        # Setup NeRF Modules
        self.nerf_encoder = nerf.encoder.to(self.device)
        self.nerf_bound: torch.Tensor
        self.register_buffer('nerf_bound', torch.tensor(nerf.bound, device=self.device))

        self.nerf_opacity_and_color_net = nerf.sigma_net.to(self.device)

        self.nerf_scale_and_quaternion_net = MLP(nerf.in_dim, 7, dim_hidden=64, num_layers=3, bias=True)
        
        self.init_scale = cfg.render.init_scale

    def reset_by_state_dict(self, state_dict: Mapping[str, Any]):
        super().reset_by_state_dict(state_dict)
        self.vertex_indices = self.initialize_vertex_indices(state_dict['_positions'], self.canonical_vertices)
        for name in ('nerf_opacities', 'nerf_scales', 'nerf_quaternions'):
            if name in state_dict:
                self.register_buffer(name, state_dict[name])

    def forward(self) -> GaussianOutput:
        positions = self.get_positions()
        opacities, colors, scales, quaternions = self.nerf_forward(positions)
        return GaussianOutput(
            positions=positions,
            opacities=opacities,
            colors=colors,
            sh_features=None,
            quaternions=quaternions,
            scales=scales,
        )
    
    def nerf_forward(self, xyzs):
        # Forward
        enc = self.nerf_encoder(xyzs, bound=self.nerf_bound)  # xyzs: [N, 3] -> enc: [N, 32]
        # Get colors and opacities
        opacities_and_colors = self.nerf_opacity_and_color_net(enc)
        
        colors = self.color_activation(opacities_and_colors[:, 1:])
        opacities = self.opacity_activation(opacities_and_colors[:, :1])
        # Get both scales and quaternions
        scales_quaternions = self.nerf_scale_and_quaternion_net(enc)
        scales = self.scale_activation(scales_quaternions[:, :3]) * self.init_scale
        quaternions = self.rotation_activation(scales_quaternions[:, 3:])
        # Return
        return opacities, colors, scales, quaternions

    def get_optimizer(self, cfg:TrainConfig):
        # Gaussians
        optimizers = super().get_optimizer(cfg=cfg)
        # NeRF
        nerf_lr = cfg.nerf.lr
        params = [{'params': self.nerf_encoder.parameters(), 'lr': nerf_lr * 10}]
        if hasattr(self, 'nerf_opacity_and_color_net'):
            params += [{'params': self.nerf_opacity_and_color_net.parameters(), 'lr': nerf_lr},]
        if hasattr(self, 'nerf_scale_and_quaternion_net'):
            params += [{'params': self.nerf_scale_and_quaternion_net.parameters(), 'lr': nerf_lr},]
        optimizers['nerf'] = torch.optim.Adam(params, betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        # Return
        return optimizers


class HashAvatarWithMesh(HashAvatar):
    def __init__(
        self,
        cfg: TrainConfig,
        predefined_vertex_indices: torch.Tensor,
        predefined_triangle_indices: torch.Tensor,
        learn_bary_coords: bool = False,
        learn_vertex_coords: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, **kwargs)
        
        self.learn_bary_coords = learn_bary_coords
        self.learn_vertex_coords = learn_vertex_coords

        self.learn_mesh_scales = cfg.render.learn_mesh_scales
        self.learn_mesh_quaternions = cfg.render.learn_mesh_quaternions
        self.use_nerf_mesh_opacities = cfg.render.use_nerf_mesh_opacities
        self.use_nerf_mesh_scales_and_quaternions = cfg.render.use_nerf_mesh_scales_and_quaternions

        if learn_bary_coords:
            self.bary_coords_activation = torch.nn.Softmax(dim=-1)
        else:
            self.bary_coords_activation = torch.nn.Identity()
        
        n_gaussians_per_triangle = cfg.render.n_gaussians_per_triangle
        if not learn_bary_coords:
            assert n_gaussians_per_triangle in (1, 3, 4, 6)

        predefined_triangles = self.canonical_triangles[predefined_triangle_indices]
        n_points_on_surface = len(predefined_triangles) * n_gaussians_per_triangle + len(predefined_vertex_indices)
        self._n_points_on_surface = n_points_on_surface

        bary_coords_per_triangle = self.prepare_mesh_binding_params(n_gaussians_per_triangle)  # [n_gaussians_per_triangle, 3]
        bary_coords = bary_coords_per_triangle.expand(len(predefined_triangles), -1, -1)  # [Nf, n_gaussians_per_triangle, 3]
        self._bary_coords = nn.Parameter(bary_coords, requires_grad=learn_bary_coords)
        triangle_coords = self.canonical_vertices[predefined_triangles]  # [Nf, 3, 3]
        self.register_buffer('_triangle_coords', triangle_coords)

        vertex_coords = self.canonical_vertices[predefined_vertex_indices]
        self._vertex_coords = nn.Parameter(vertex_coords, requires_grad=learn_vertex_coords)

        positions_on_surface = self.get_mesh_positions()
        vertex_indices_on_surface = knn_points(positions_on_surface[None], self.canonical_vertices[None], K=1).idx[0, :, 0]
        self.vertex_indices_on_surface = vertex_indices_on_surface
        self.densification_mask = torch.cat([
            self.densification_mask,
            torch.zeros_like(self.vertex_indices_on_surface, dtype=torch.bool),
        ])

        if not self.use_nerf_mesh_scales_and_quaternions:
            # Initialize scales
            scales_on_surface = self.initialize_scales(
                cfg=cfg,
                n_points=self._n_points_on_surface,
                vertices=self.canonical_vertices,
                vertex_indices=self.vertex_indices_on_surface,
            )
            self._scales_on_surface = nn.Parameter(scales_on_surface, requires_grad=self.learn_mesh_scales)
            # Initialize quaternions
            quaternions_on_surface = matrix_to_quaternion(torch.eye(3)[None].repeat(self._n_points_on_surface, 1, 1))  # [N, 4]
            self._quaternions_on_surface = nn.Parameter(quaternions_on_surface, requires_grad=self.learn_mesh_quaternions)

        # Buffer Mechanism for mesh-binding 3D gaussians
        self.buffer_mesh_opacities_from_nerf = cfg.render.buffer_mesh_opacities_from_nerf
        self.buffer_mesh_scales_from_nerf = cfg.render.buffer_mesh_scales_from_nerf
        self.buffer_mesh_quaternions_from_nerf = cfg.render.buffer_mesh_quaternions_from_nerf
        self.use_mesh_opacities_buffer = cfg.render.use_mesh_opacities_buffer
        self.use_mesh_scales_buffer = cfg.render.use_mesh_scales_buffer
        self.use_mesh_quaternions_buffer = cfg.render.use_mesh_quaternions_buffer

        self.requires_buffer_on_surface = \
            self.buffer_mesh_opacities_from_nerf or \
            self.buffer_mesh_scales_from_nerf or \
            self.buffer_mesh_quaternions_from_nerf

    def get_vertex_indices(self) -> torch.Tensor:
        return torch.cat([self.vertex_indices, self.vertex_indices_on_surface], dim=0)

    def prepare_mesh_binding_params(self, n_gaussians_per_triangle) -> torch.Tensor:

        if n_gaussians_per_triangle == 1:
            bary_coords_per_triangle = torch.tensor([
                [1/3, 1/3, 1/3],
            ], dtype=torch.float32)

        elif n_gaussians_per_triangle == 3:
            bary_coords_per_triangle = torch.tensor([
                [1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]
            ], dtype=torch.float32)

        elif n_gaussians_per_triangle == 4:
            bary_coords_per_triangle = torch.tensor([
                [1/3, 1/3, 1/3],
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3]
            ], dtype=torch.float32)

        elif n_gaussians_per_triangle == 6:
            bary_coords_per_triangle = torch.tensor([
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
                [1/6, 5/12, 5/12],
                [5/12, 1/6, 5/12],
                [5/12, 5/12, 1/6]
            ], dtype=torch.float32)
        
        else:
            bary_coords_per_triangle = torch.rand(n_gaussians_per_triangle, 3, dtype=torch.float32)
            bary_coords_per_triangle = F.softmax(bary_coords_per_triangle, dim=1)
        
        return bary_coords_per_triangle

    def reset_by_state_dict(self, state_dict: Mapping[str, Any]):
        super().reset_by_state_dict(state_dict=state_dict)
        with torch.no_grad():
            positions_on_surface = self.get_mesh_positions(
                _bary_coords=state_dict['_bary_coords'],
                _triangle_coords=state_dict['_triangle_coords'],
                _vertex_coords=state_dict['_vertex_coords'],
            )
            vertex_indices_on_surface = knn_points(
                positions_on_surface[None].detach().cpu(),
                self.canonical_vertices[None].detach().cpu(),
                K=1,
            ).idx[0, :, 0]
        self.vertex_indices_on_surface = vertex_indices_on_surface
        self.densification_mask = torch.cat([
            self.densification_mask,
            torch.zeros_like(self.vertex_indices_on_surface, dtype=torch.bool),
        ])
        for name in ('nerf_opacities_on_surface', 'nerf_scales_on_surface', 'nerf_quaternions_on_surface'):
            if name in state_dict:
                self.register_buffer(name, state_dict[name])

    def get_mesh_positions(self, _bary_coords=None, _triangle_coords=None, _vertex_coords=None):
        
        if _bary_coords is None:
            _bary_coords = self._bary_coords
        if _triangle_coords is None:
            _triangle_coords = self._triangle_coords
        if _vertex_coords is None:
            _vertex_coords = self._vertex_coords
        
        # _bary_coords: [n_faces, n_gaussians_per_face, n_vertices_per_face]
        # _triangle_coords: [n_faces, n_vertices_per_face, n_coords]
        # positions_on_triangles: [n_faces, n_gaussians_per_face, n_coords]
        bary_coords = self.bary_coords_activation(_bary_coords)
        positions_on_triangles = torch.einsum('fnv,fvc->fnc', bary_coords, _triangle_coords)
        positions_on_triangles = positions_on_triangles.reshape(-1, 3)
        
        positions_on_vertices = _vertex_coords
        
        return torch.cat([positions_on_vertices, positions_on_triangles], dim=0)

    def get_mesh_scales(self, return_means=False):
        if not return_means:
            return self.scale_activation(self._scales_on_surface)
        else:
            return self.scale_activation(self._scales_on_surface.mean(dim=-1, keepdim=True).expand(-1, 3))
    
    def get_mesh_quaternions(self):
        return self.rotation_activation(self._quaternions_on_surface)

    def buffer_mesh_attributes_from_nerf(self, positions_on_surface):
        with torch.no_grad():
            opacities_on_surface, colors_on_surface, scales_on_surface, quaternions_on_surface = \
                self.nerf_mesh_forward(positions_on_surface, disable_buffer=True)
        if self.buffer_mesh_opacities_from_nerf:
            self.register_buffer('nerf_opacities_on_surface', opacities_on_surface)
        if self.buffer_mesh_scales_from_nerf:
            self.register_buffer('nerf_scales_on_surface', scales_on_surface)
        if self.buffer_mesh_quaternions_from_nerf:
            self.register_buffer('nerf_quaternions_on_surface', quaternions_on_surface)

    def nerf_mesh_forward(self, xyzs, disable_buffer:bool=False):
        # Buffer Mechanism
        if self.requires_buffer_on_surface:
            self.requires_buffer_on_surface = False
            self.buffer_mesh_attributes_from_nerf(xyzs.detach())
        # Forward
        enc = self.nerf_encoder(xyzs, bound=self.nerf_bound)
        # Get colors and opacities
        opacities_and_colors = self.nerf_opacity_and_color_net(enc)
        colors = self.color_activation(opacities_and_colors[:, 1:])
        if self.use_nerf_mesh_opacities:
            if self.use_mesh_opacities_buffer and not disable_buffer:
                opacities = self.nerf_opacities_on_surface
            else:
                opacities = opacities_and_colors[:, :1]
                opacities = self.opacity_activation(opacities)
        else:
            opacities = torch.ones((self._n_points_on_surface, 1), device=self.device)
        # Get both scales and quaternions
        if self.use_nerf_mesh_scales_and_quaternions:
            if self.use_mesh_scales_buffer and self.use_mesh_quaternions_buffer and not disable_buffer:
                scales = self.nerf_scales_on_surface
                quaternions = self.nerf_quaternions_on_surface
            else:
                scales_quaternions = self.nerf_scale_and_quaternion_net(enc)
                if self.use_mesh_scales_buffer and not disable_buffer:
                    scales = self.nerf_scales_on_surface
                else:
                    scales = self.scale_activation(scales_quaternions[:, :3]) * self.init_scale
                if self.use_mesh_quaternions_buffer and not disable_buffer:
                    quaternions = self.nerf_quaternions_on_surface
                else:
                    quaternions = self.rotation_activation(scales_quaternions[:, 3:])
        else:
            scales = self.get_mesh_scales()
            quaternions = self.get_mesh_quaternions()
        # Return
        return opacities, colors, scales, quaternions

    def forward(self) -> GaussianOutput:
        positions = self.get_positions()
        opacities, colors, scales, quaternions = self.nerf_forward(positions)

        positions2 = self.get_mesh_positions()
        opacities2, colors2, scales2, quaternions2 = self.nerf_mesh_forward(positions2)

        return GaussianOutput(
            positions=torch.cat([positions, positions2], dim=0),
            opacities=torch.cat([opacities, opacities2], dim=0),
            colors=torch.cat([colors, colors2], dim=0),
            sh_features=None,
            quaternions=torch.cat([quaternions, quaternions2], dim=0),
            scales=torch.cat([scales, scales2], dim=0),
        )
    
    def get_optimizer(self, cfg:TrainConfig):
        optimizers = super().get_optimizer(cfg=cfg)
        l = []
        if self.learn_bary_coords:
            l += [{'params': [self._bary_coords], 'lr': cfg.render.position_lr_init, 'name': "bary_coords"}]
        if self.learn_vertex_coords:
            l += [{'params': [self._vertex_coords], 'lr': cfg.render.position_lr_init, 'name': "vertex_coords"}]
        if not self.use_nerf_mesh_scales_and_quaternions:
            if self.learn_mesh_scales:
                l += [{'params': [self._scales_on_surface], 'lr': cfg.render.scaling_lr, 'name': "scales_on_surface"}]
            if self.learn_mesh_quaternions:
                l += [{'params': [self._quaternions_on_surface], 'lr': cfg.render.rotation_lr, 'name': "quaternions_on_surface"}]
        if len(l) > 0:
            optimizers['mesh'] = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizers


####################################################################################
def find_knn_vertex_indices(positions: torch.Tensor, vertices: torch.Tensor, K: int = 1):
    vertex_indices = knn_points(
        positions[None].data.detach(),
        vertices[None].detach(),
        K=K,
    ).idx[0]  # [N, K]
    assert vertex_indices.shape[1] == K
    if K == 1:
        return vertex_indices[:, 0]
    else:
        return vertex_indices

def find_nearest_triangles(points, vertices, triangles, device=None):
    # Define the function to find nearest triangle and compute barycentric coordinates
    import igl
    if isinstance(points, nn.Parameter):
        points = points.data.detach().cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(triangles, torch.Tensor):
        triangles = triangles.detach().cpu().numpy()

    # closest_faces: np.ndarray [N,]
    # squared_distances: np.ndarray [N,]
    # triangle_indices: np.ndarray [N,]
    # closest_points: np.ndarray [N, 3]
    squared_distances, triangle_indices, closest_points = igl.point_mesh_squared_distance(points, vertices, triangles)

    vertex_indices = triangles[triangle_indices]  # [N, 3]
    vertex_coords = vertices[vertex_indices]  # [N, 3, 3]

    v0 = np.ascontiguousarray(vertex_coords[:, 0, :])
    v1 = np.ascontiguousarray(vertex_coords[:, 1, :])
    v2 = np.ascontiguousarray(vertex_coords[:, 2, :])
    barycentric_coords = igl.barycentric_coordinates_tri(closest_points, v0, v1, v2)  # [N, 3]

    triangle_indices = torch.tensor(triangle_indices, device='cpu')
    squared_distances = torch.tensor(squared_distances, device=device)
    barycentric_coords = torch.tensor(barycentric_coords, device=device)
    vertex_indices = torch.tensor(vertex_indices, device='cpu')

    min_barycentric_indices = torch.argmin(barycentric_coords, dim=1, keepdim=True)  # [N, 1]
    nearest_vertex_indices = torch.gather(vertex_indices, 1, min_barycentric_indices).squeeze(-1)  # [N,]

    return {
        'squared_distances': squared_distances,
        'triangle_indices': triangle_indices,
        'vertex_indices': vertex_indices,
        'nearest_vertex_indices': nearest_vertex_indices,
        'barycentric_coords': barycentric_coords,
    }

def prune_points_close_to_mesh(
    positions: torch.Tensor,
    nearest_triangles_buffer: dict,
    predefined_triangle_indices: torch.Tensor,
    threshold: Optional[float] = None,
):
    n1 = positions.shape[0]

    squared_distances = nearest_triangles_buffer['squared_distances']
    triangle_indices = nearest_triangles_buffer['triangle_indices']
    prune_mask = torch.isin(triangle_indices, predefined_triangle_indices)
    if threshold is not None:
        prune_mask &= (squared_distances < threshold ** 2)

    positions = positions[~prune_mask]
    for k in nearest_triangles_buffer.keys():
        nearest_triangles_buffer[k] = nearest_triangles_buffer[k][~prune_mask]

    n2 = positions.shape[0]

    logger.info(f'Pruned points close to the mesh-binding points: {n1} -> {n2}')

    return positions, nearest_triangles_buffer

def remapping(x: np.ndarray, keys: np.ndarray, values=None):
    """
    Remaps the values in array `x` based on the given `keys` and `values`.

    Parameters:
    x (np.ndarray): The input array to be remapped.
    keys (np.ndarray): The keys used for remapping.
    values (np.ndarray, optional): The values used for remapping. If not provided, the values will be generated using `np.arange(len(keys))`.

    Returns:
    np.ndarray: The remapped array.

    Raises:
    AssertionError: If the input arrays have incorrect data types or dimensions.

    """
    assert np.issubdtype(x.dtype, np.integer)
    assert np.issubdtype(keys.dtype, np.integer)
    assert keys.ndim == 1
    if values is None:
        values = np.arange(len(keys))
    else:
        assert values.ndim == 1
        assert np.issubdtype(values.dtype, np.integer)
    mappings = np.column_stack((keys, values))
    mappings = mappings[np.argsort(mappings[:, 0])]
    indices = np.searchsorted(mappings[:, 0], x)
    return mappings[indices, 1]


class LBSUtils:
    @staticmethod
    @torch.no_grad()
    def initialize_lbs_weights(
        lbs_model: GeneralLinearBlendSkinning,
        nearest_triangles_buffer: dict,
        positions: torch.Tensor = None,
        smooth: bool = False,
        smooth_K: int = None,
        smooth_N: int = None,
        use_sqrt: bool = True,
        valid_dist_threshold: float = 0.01,
    ) -> torch.Tensor:
        lbs_weights = lbs_model.lbs_weights[nearest_triangles_buffer['vertex_indices']]  # [N, 3, J]
        barycentric_coords = nearest_triangles_buffer['barycentric_coords'].to(lbs_weights.device)  # [N, 3]
        lbs_weights = torch.einsum('nij,ni->nj', lbs_weights, barycentric_coords)  # [N, J]
        # Nearest Neighbor Smoothing
        if smooth:
            # n_points = positions.shape[0]
            # 2423877 -> K=30, N=1000
            logger.info(f'Using K={smooth_K}, N={smooth_N} for LBS weight smoothing')
            knn_results = knn_points(positions[None], positions[None], K=smooth_K+1, device=lbs_weights.device)
            knn_indices = knn_results.idx[0][:, 1:]  # [N, K]
            knn_dists = knn_results.dists[0][:, 1:]  # [N, K]
            mesh_dists = nearest_triangles_buffer['squared_distances'].to(lbs_weights.device)  # [N,]
            if use_sqrt:
                knn_dists = torch.sqrt(knn_dists)
                mesh_dists = torch.sqrt(mesh_dists)
            knn_weights = 1.0 / (mesh_dists[knn_indices] * knn_dists)  # [N, K]
            knn_weights = knn_weights / knn_weights.sum(dim=-1, keepdim=True)  # [N, K]
            def dists_to_weights(dists:torch.Tensor, low:float=None, high:float=None):
                if low is None: low = high
                if high is None: high = low
                assert high >= low
                weights = dists.clone()
                weights[dists <= low] = 0.0
                weights[dists >= high] = 1.0
                indices = (dists > low) & (dists < high)
                weights[indices] = (dists[indices] - low) / (high - low)
                return weights
            update_weights = dists_to_weights(mesh_dists, low=valid_dist_threshold).unsqueeze(-1)  # [N, 1]
            # update_weights = None
            for _ in range(smooth_N):
                new_lbs_weights = torch.einsum('nk,nkj->nj', knn_weights, lbs_weights[knn_indices])
                if update_weights is None:
                    lbs_weights = new_lbs_weights
                else:
                    lbs_weights = (1.0 - update_weights) * lbs_weights + update_weights * new_lbs_weights

        return lbs_weights

    @staticmethod
    def lbs_weight_activation(lbs_weights):
        return lbs_weights / lbs_weights.sum(dim=-1, keepdim=True)

    def get_lbs_weights(self):
        return self.lbs_weight_activation(self._lbs_weights)


class MeshBindingGaussianModel(nn.Module, LBSUtils):
    def __init__(
        self,
        cfg: TrainConfig,
        v_template: torch.Tensor,
        canonical_triangles: torch.Tensor,
        predefined_vertex_indices: List,
        predefined_triangle_indices: List,
        learn_bary_coords: bool = True,
        learn_vertex_coords: bool = False,
        learn_scales: bool = True,
        init_scale_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.learn_bary_coords = learn_bary_coords
        self.learn_vertex_coords = learn_vertex_coords
        self.learn_scales = learn_scales

        self.predefined_vertex_indices = torch.tensor(predefined_vertex_indices)
        self.predefined_triangle_indices = torch.tensor(predefined_triangle_indices)

        self._n_points_per_triangle = cfg.render.n_gaussians_per_triangle
        self._n_triangles = len(predefined_triangle_indices)
        self._n_vertices = len(predefined_vertex_indices)
        self._n_points = self._n_triangles * self._n_points_per_triangle

        bary_coords = self.initialize_bary_coords(self._n_points_per_triangle)  # [F, G_per_F, 3]
        self._bary_coords = nn.Parameter(bary_coords, requires_grad=learn_bary_coords)
        
        vertex_coords = v_template[predefined_vertex_indices]
        self._vertex_coords = nn.Parameter(vertex_coords, requires_grad=learn_vertex_coords)

        predefined_triangles = canonical_triangles[predefined_triangle_indices].cpu().numpy()
        triangles = remapping(predefined_triangles, np.array(predefined_vertex_indices))
        self.register_buffer('triangles', torch.tensor(triangles))
        self.triangles: torch.LongTensor

        points_to_triangles = torch.arange(self._n_triangles)[...,None].expand(-1, self._n_points_per_triangle).reshape(-1)  # [N,]
        points_to_vertices = self.triangles[points_to_triangles]  # [N, 3]
        self.register_buffer('points_to_triangles', points_to_triangles)
        self.register_buffer('points_to_vertices', points_to_vertices)
        self.points_to_triangles: torch.LongTensor
        self.points_to_vertices: torch.LongTensor

        self._scales = nn.Parameter(torch.ones(self._n_points, 3) * init_scale_ratio, requires_grad=learn_scales)

    def initialize_bary_coords(self, n_gaussians_per_triangle) -> torch.Tensor:

        if not self.learn_bary_coords:
            assert n_gaussians_per_triangle in (1, 3, 4, 6)

        if n_gaussians_per_triangle == 1:
            bary_coords_per_triangle = torch.tensor([
                [1/3, 1/3, 1/3],
            ], dtype=torch.float32)

        elif n_gaussians_per_triangle == 3:
            bary_coords_per_triangle = torch.tensor([
                [1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]
            ], dtype=torch.float32)

        elif n_gaussians_per_triangle == 4:
            bary_coords_per_triangle = torch.tensor([
                [1/3, 1/3, 1/3],
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3]
            ], dtype=torch.float32)

        elif n_gaussians_per_triangle == 6:
            bary_coords_per_triangle = torch.tensor([
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
                [1/6, 5/12, 5/12],
                [5/12, 1/6, 5/12],
                [5/12, 5/12, 1/6]
            ], dtype=torch.float32)
        
        else:
            bary_coords_per_triangle = torch.rand(n_gaussians_per_triangle, 3, dtype=torch.float32)
            bary_coords_per_triangle = self.bary_coord_activation(bary_coords_per_triangle)
        
        return bary_coords_per_triangle.expand(self._n_triangles, -1, -1)

    @staticmethod
    def bary_coord_activation(bary_coords):
        return bary_coords / bary_coords.sum(dim=-1, keepdim=True)

    def get_vertex_coords(self):
        return self._vertex_coords

    def get_positions(self, vertex_coords=None, bary_coords=None):
        if bary_coords is None:
            bary_coords = self.bary_coord_activation(self._bary_coords)  # [F, G_per_F, 3]

        if vertex_coords is None:
            vertex_coords = self.get_vertex_coords()
        
        triangle_coords = vertex_coords[self.triangles]  # [F, V_per_F=3, 3]
        
        return torch.einsum('fnv,fvc->fnc', bary_coords, triangle_coords).reshape(-1, 3)

    def get_scales_and_quaternions(self, vertex_coords, positions, eps=1e-9):
        """
            approximate covariance matrix and calculate scaling/rotation tensors

            covariance matrix is [v0, v1, v2], where
            v0 is a normal vector to each face
            v1 is a vector from centroid of each face and 1st vertex
            v2 is obtained by orthogonal projection of a vector from centroid to 2nd vertex onto subspace spanned by v0 and v1
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        p0 = positions  # [N, 3]
        p1, p2, p3 = torch.split(vertex_coords[self.points_to_vertices], 1, dim=1)
        p1, p2, p3 = p1.squeeze(1), p2.squeeze(1), p3.squeeze(1)  # [N, 3], [N, 3], [N, 3]

        vertex_normals, _ = compute_normal(vertex_coords, self.triangles)  # [V, 3]

        point_vertex_normals = vertex_normals[self.points_to_vertices]  # [N, 3, 3]
        point_bary_coords = self._bary_coords.reshape(-1, 3)[:, :, None]                 # [N, 3, 1]
        point_normals = (point_vertex_normals * point_bary_coords).sum(dim=1)            # [N, 3]

        v0 = point_normals / (torch.linalg.vector_norm(point_normals, dim=-1, keepdim=True) + eps)
        
        reference = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=positions.device).expand_as(p0)
        v1 = torch.cross(v0, reference, dim=1)
        v1 = v1 / (torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps)
        v2 = torch.cross(v0, v1, dim=1)
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        rotation_matrices = torch.stack((v0, v1, v2), dim=2)
        rotation_matrices[:, [1, 2], :] *= -1  # Fix the direction of the axis

        s0 = torch.zeros_like(v0[:, [0]])
        s1 = (dot(p1 - p0, v1).abs() + dot(p2 - p0, v1).abs() + dot(p3 - p0, v1).abs()) / self._n_points_per_triangle
        s2 = (dot(p1 - p0, v2).abs() + dot(p2 - p0, v2).abs() + dot(p3 - p0, v2).abs()) / self._n_points_per_triangle
        s1 *= torch.clamp(self._scales[:, [1]], min=0.5, max=2.0)
        s2 *= torch.clamp(self._scales[:, [2]], min=0.5, max=2.0)
        scales = torch.concat((s0, s1, s2), dim=1)

        quaternions = standardize_quaternion(matrix_to_quaternion(rotation_matrices))

        return scales, quaternions

    def get_optimizer(self, cfg: TrainConfig, optimizer_name: str):
        optimizers = {}
        l = []
        if self.learn_bary_coords:
            l += [{'params': [self._bary_coords], 'lr': cfg.render.position_lr_init, 'name': "bary_coords"}]
        if self.learn_vertex_coords:
            l += [{'params': [self._vertex_coords], 'lr': cfg.render.position_lr_init, 'name': "vertex_coords"}]
        if self.learn_scales:
            l += [{'params': [self._scales], 'lr': cfg.render.scaling_lr, 'name': "scales"}]
        if len(l) > 0:
            optimizers[optimizer_name] = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            logger.warning('No mesh-binding 3D gaussian parameters to optimize')
        return optimizers


class DreamWaltzG(_AnimatableAvatar, LBSUtils):
    def __init__(
        self,
        cfg: TrainConfig,
        nerf: NeRFNetwork,
        predefined_meshes: dict,
        point_cloud: BasicPointCloud = None,
        learn_positions: bool = True,
        learn_scales: bool = False,
        learn_quaternions: bool = False,
        learn_lbs_weights: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the Avatar class.

        """
        super().__init__(cfg=cfg, **kwargs)
        assert type(self.lbs_model) is GeneralLinearBlendSkinning

        # Initialize args
        self.use_joint_shape_offsets = cfg.render.use_joint_shape_offsets
        self.use_vertex_shape_offsets = cfg.render.use_vertex_shape_offsets
        self.use_vertex_pose_offsets = cfg.render.use_vertex_pose_offsets

        self.use_non_rigid_offsets = cfg.render.use_non_rigid_offsets
        self.use_non_rigid_scales = cfg.render.use_non_rigid_scales
        self.use_non_rigid_rotations = cfg.render.use_non_rigid_rotations

        self.non_rigid_scale_mode = cfg.render.non_rigid_scale_mode
        self.non_rigid_rotation_mode = cfg.render.non_rigid_rotation_mode

        self.render_mesh_binding_3d_gaussians_only = cfg.render.render_mesh_binding_3d_gaussians_only
        self.render_unconstrained_3d_gaussians_only = cfg.render.render_unconstrained_3d_gaussians_only

        assert not (self.use_joint_shape_offsets and self.use_vertex_shape_offsets)

        # Initialize mesh-binding parameters
        n_points_on_mesh = 0
        mesh_binding_gaussians = nn.ModuleDict()
        for k, v in predefined_meshes.items():
            mesh_binding_gaussians[k] = MeshBindingGaussianModel(
                cfg=cfg,
                v_template=self.lbs_model.v_template.data.detach(),
                canonical_triangles=self.canonical_triangles,
                predefined_vertex_indices=v['vertex_indices'],
                predefined_triangle_indices=v['triangle_indices'],
                learn_bary_coords=cfg.render.learn_mesh_bary_coords,
                learn_vertex_coords=cfg.render.learn_mesh_vertex_coords,
                learn_scales=cfg.render.learn_mesh_scales,
            )
            n_points_on_mesh += mesh_binding_gaussians[k]._n_points
        self.mesh_binding_gaussians = mesh_binding_gaussians
        self._n_points_on_mesh = n_points_on_mesh

        # Setup NeRF modules
        self.nerf_encoder = nerf.encoder.to(self.device)
        self.nerf_bound: torch.Tensor
        self.register_buffer('nerf_bound', torch.tensor(nerf.bound, device=self.device))
        self.nerf_opacity_and_color_net = nerf.sigma_net.to(self.device)

        if cfg.render.reset_nerf:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)
            self.nerf_opacity_and_color_net.apply(init_weights)
        
        self.use_nerf_encoded_position = cfg.render.use_nerf_encoded_position
        self.init_scale = cfg.render.init_scale
        self.max_scale = cfg.render.max_scale
        self.init_offset = cfg.render.init_offset

        if self.use_nerf_encoded_position:
            self.nerf_scale_and_quaternion_net = DeformNetwork(xyz_input_ch=32, D=4, W=64, residual=False)
        else:
            self.nerf_scale_and_quaternion_net = DeformNetwork(xyz_input_ch=None, D=4, W=64, residual=False)

        # Initialize positions
        positions = self.initialize_positions(
            cfg=cfg,
            vertices=self.canonical_vertices,
            triangles=self.canonical_triangles,
            point_cloud=point_cloud,
        )

        # Initialize vertex indices and densification mask
        nearest_triangles_buffer = find_nearest_triangles(
            points=positions,
            vertices=self.canonical_vertices,
            triangles=self.canonical_triangles,
        )
        
        # Prune points close to the mesh-binding points
        if point_cloud is not None and cfg.render.prune_points_close_to_mesh:
            for body_part, m in self.mesh_binding_gaussians.items():
                m: MeshBindingGaussianModel
                if body_part == 'hands':
                    threshold = cfg.render.prune_dists_close_to_mesh * 10
                else:
                    assert body_part == 'face'
                    threshold = cfg.render.prune_dists_close_to_mesh
                positions, nearest_triangles_buffer = prune_points_close_to_mesh(
                    positions,
                    nearest_triangles_buffer=nearest_triangles_buffer,
                    predefined_triangle_indices=m.predefined_triangle_indices,
                    threshold=threshold,
                )

        # Initialize nearest_triangles_buffer
        self.nearest_triangles_buffer = nearest_triangles_buffer

        # Initialize LBS weights
        lbs_weights = self.initialize_lbs_weights(
            lbs_model=self.lbs_model,
            nearest_triangles_buffer=self.nearest_triangles_buffer,
            positions=positions,
            smooth=cfg.render.lbs_weight_smooth,
            smooth_K=cfg.render.lbs_weight_smooth_K,
            smooth_N=cfg.render.lbs_weight_smooth_N,
        )
        self._lbs_weights = nn.Parameter(lbs_weights, requires_grad=learn_lbs_weights)
        self.learn_lbs_weights = learn_lbs_weights

        self.learn_hand_betas = cfg.render.learn_hand_betas
        self.learn_face_betas = cfg.render.learn_face_betas
        self.learn_betas = self.learn_hand_betas or self.learn_face_betas
        self._betas = nn.Parameter(self.lbs_model.betas.data.clone(), requires_grad=self.learn_betas)
        
        # Initialize positions
        with torch.no_grad():
            positions = self.inverse_lbs_transform(
                positions=positions.to(self.device),
                transforms=self.lbs_model.forward(**self.smpl_canonical_inputs)[-1],
            )
        self._positions = nn.Parameter(positions, requires_grad=learn_positions)
        self.learn_positions = learn_positions
        self._n_points = len(positions)

        # Initialize scales
        scales = self.scale_inverse_activation(torch.ones(self._n_points, 3) * self.init_scale)
        self._scales = nn.Parameter(scales, requires_grad=learn_scales)
        self.learn_scale = learn_scales

        # Initialize quaternions
        quaternions = matrix_to_quaternion(torch.eye(3)[None].repeat(self._n_points, 1, 1))  # [N, 4]
        self._quaternions = nn.Parameter(quaternions, requires_grad=learn_quaternions)
        self.learn_quaternions = learn_quaternions

    @property
    def densification_mask(self):
        return torch.cat([
            torch.ones(self._n_points, dtype=torch.bool),
            torch.zeros(self._n_points_on_mesh, dtype=torch.bool),
        ])

    def reset_by_state_dict(self, state_dict: Mapping[str, Any]):
        attribute_names = [
            '_positions',
            '_sh_features_dc',
            '_sh_features_rest',
            '_scales',
            '_quaternions',
            '_opacities',
        ]
        self.nearest_triangles_buffer = find_nearest_triangles(
            points=state_dict['_positions'],
            vertices=self.canonical_vertices,
            triangles=self.canonical_triangles,
        )
        if hasattr(self, '_lbs_weights'):
            if '_lbs_weights' not in state_dict:
                lbs_weights = self.initialize_lbs_weights(
                    lbs_model=self.lbs_model,
                    nearest_triangles_buffer=self.nearest_triangles_buffer,
                    positions=state_dict['_positions'],
                    smooth=self.cfg.render.lbs_weight_smooth,
                    smooth_K=self.cfg.render.lbs_weight_smooth_K,
                    smooth_N=self.cfg.render.lbs_weight_smooth_N,
                )
                self._lbs_weights = nn.Parameter(lbs_weights, requires_grad=self.learn_lbs_weights)
            else:
                attribute_names += ['_lbs_weights']
        super().reset_by_state_dict(state_dict, attribute_names=attribute_names)
        
    def static_mlp_forward(self, enc: torch.Tensor, fix_opacities: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        opacities_and_colors = self.nerf_opacity_and_color_net(enc)
        colors = self.color_activation(opacities_and_colors[:, 1:])
        if fix_opacities:
            opacities = torch.ones_like(opacities_and_colors[:, :1])
        else:
            opacities = self.opacity_activation(opacities_and_colors[:, :1])
        return colors, opacities

    def dynamic_mlp_forward(self, enc: torch.Tensor, body_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nerf_offsets, nerf_scales, nerf_quaternions = self.nerf_scale_and_quaternion_net(enc, body_pose)
        return nerf_offsets, nerf_scales, nerf_quaternions

    def get_unconstrained_gaussians(
        self,
        positions: torch.Tensor,
        canonical_positions: torch.Tensor,
        smpl_observed_inputs: dict,
        fix_opacities: bool = False,
    ) -> GaussianOutput:
        # Get colors and opacities
        enc = self.nerf_encoder(canonical_positions, bound=self.nerf_bound)  # positions: [N, 3] -> enc: [N, 32]
        colors, opacities = self.static_mlp_forward(enc, fix_opacities=fix_opacities)

        # Get scales and quaternions
        if 'body_pose' in smpl_observed_inputs:
            body_pose = smpl_observed_inputs['body_pose']
        else:
            body_pose = torch.zeros((1, 63), device=self.device)
        
        if self.use_nerf_encoded_position:
            offsets, scales, quaternions = self.dynamic_mlp_forward(enc, body_pose=body_pose)
        else:
            offsets, scales, quaternions = self.dynamic_mlp_forward(positions.detach(), body_pose=body_pose)
        
        # Construct GaussianOutput
        return GaussianOutput(
            positions=positions,
            opacities=opacities,
            colors=colors,
            quaternions=quaternions,
            scales=scales,
            offsets=offsets,
        )

    def get_constrained_gaussians(
        self,
        positions: torch.Tensor,
        use_nerf: bool = True,
        mean_color: bool = False,
        mean_color_rate: float = 0.5,
        fix_opacities: bool = True,
    ) -> GaussianOutput:

        # Mesh-Independent Forward
        if use_nerf:
            enc = self.nerf_encoder(positions, bound=self.nerf_bound)  # positions: [N, 3] -> enc: [N, 32]
            colors, opacities = self.static_mlp_forward(enc, fix_opacities=fix_opacities)
            if mean_color:
                n_points = colors.size(0)
                mean_colors = torch.mean(colors, dim=0, keepdim=True).expand(n_points, -1)
                colors = mean_color_rate * mean_colors + (1.0 - mean_color_rate) * colors
        else:
            assert 0

        # Return
        return GaussianOutput(
            positions=None,
            opacities=opacities,
            colors=colors,
            quaternions=None,
            scales=None,
        )

    def get_opacities(self, return_ones: bool = False) -> torch.Tensor:
        # Get uncontrained gaussians
        positions = self.get_positions()
        lbs_weights = self.get_lbs_weights()
        vertex_indices = self.nearest_triangles_buffer['nearest_vertex_indices']

        # Get LBS Transform from smpl-canonical to canonical space
        _, _, canonical_transforms = self.lbs_model.forward(**self.smpl_canonical_inputs)

        # Canonical LBS Transform
        canonical_positions = self.lbs_transform(
            positions=positions,
            transforms=canonical_transforms,
            lbs_weights=lbs_weights,
            vertex_indices=vertex_indices,
        )

        # Get colors and opacities
        enc = self.nerf_encoder(canonical_positions, bound=self.nerf_bound)  # positions: [N, 3] -> enc: [N, 32]
        colors, opacities = self.static_mlp_forward(enc, fix_opacities=return_ones)

        return opacities

    def inverse_lbs_transform(
        self,
        positions: torch.Tensor,
        transforms: Dict[str, RigidTransform],
    ) -> torch.Tensor:
        # Inverse LBS Transform
        lbs_weights = self.get_lbs_weights()
        vertex_indices = self.nearest_triangles_buffer['nearest_vertex_indices']

        # Pose transforms
        # Be careful! The LBS-weighted rigid transform is not a standard SE3 transformation and cannot be invertible!
        if 0:  # Wrong
            inverse_joint_transform = RigidTransform.compose(
                transforms['J_pose_rigid'],
                transforms['G_transl_offset'],
            ).squeeze(0).weight(lbs_weights).inverse()
            positions = inverse_joint_transform.transform_points(positions)
        elif 0:  # Wrong
            inverse_joint_transform = RigidTransform.compose(
                transforms['J_pose_rigid'],
                transforms['G_transl_offset'],
            ).inverse().squeeze(0)
            positions = inverse_joint_transform.transform_points(positions, weights=lbs_weights)
        else:  # Correct
            forward_joint_transform = RigidTransform.compose(
                transforms['J_pose_rigid'],
                transforms['G_transl_offset'],
            ).squeeze(0).weight(lbs_weights)
            R_forward, T_forward = forward_joint_transform.R, forward_joint_transform.T
            positions = RigidTransform._inverse_transform_points(positions, R=R_forward, T=T_forward)
        
        # Pose offsets
        if self.use_vertex_pose_offsets:
            inverse_pose_transform: RigidTransform = transforms['V_pose_offset'].inverse().squeeze(0)
            positions = inverse_pose_transform.transform_points(positions, indices=vertex_indices)

        # Shape offsets
        if self.use_vertex_shape_offsets:
            inverse_shape_transform: RigidTransform = transforms['V_shape_offset'].inverse().squeeze(0)
            positions = inverse_shape_transform.transform_points(positions, indices=vertex_indices)
        elif self.use_joint_shape_offsets:
            inverse_shape_transform: RigidTransform = transforms['J_shape_offset'].inverse().squeeze(0)
            positions = inverse_shape_transform.transform_points(positions, weights=lbs_weights)
        
        return positions

    def lbs_transform(
        self,
        positions: torch.Tensor,
        transforms: Dict[str, RigidTransform],
        lbs_weights: torch.Tensor,
        vertex_indices: torch.Tensor,
        quaternions: Optional[torch.Tensor] = None,
    ) -> (tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        # shape offsets
        if self.use_vertex_shape_offsets:
            vertex_shape_transform: RigidTransform = transforms['V_shape_offset'].squeeze(0)
            positions = vertex_shape_transform.transform_points(positions, indices=vertex_indices)
        elif self.use_joint_shape_offsets:
            joint_shape_transform: RigidTransform = transforms['J_shape_offset'].squeeze(0)
            positions = joint_shape_transform.transform_points(positions, weights=lbs_weights)
        # pose offsets
        if self.use_vertex_pose_offsets:
            vertex_pose_transform: RigidTransform = transforms['V_pose_offset'].squeeze(0)
            positions = vertex_pose_transform.transform_points(positions, indices=vertex_indices)
        # pose transforms
        joint_pose_transform = RigidTransform.compose(
            transforms['J_pose_rigid'],
            transforms['G_transl_offset'],
        ).squeeze(0)
        positions = joint_pose_transform.transform_points(
            positions,
            weights=lbs_weights,
        )
        if quaternions is not None:
            quaternions = joint_pose_transform.transform_quaternions(
                quaternions,
                weights=lbs_weights,
                flip_rotation_axis=True,
            )
            return positions, quaternions
        else:
            return positions

    def non_rigid_transform(self, gaussians: GaussianOutput) -> GaussianOutput:
        if self.use_non_rigid_offsets:
            gaussians.positions = gaussians.positions + gaussians.offsets * self.init_offset
            gaussians.offsets = None
        
        if self.use_non_rigid_scales:
            if self.learn_scale:
                if self.non_rigid_rotation_mode == 'add':
                    gaussians.scales = self.get_scales() + gaussians.scales * self.init_scale
                else:
                    gaussians.scales = self.get_scales() * (1.0 + gaussians.scales * self.init_scale)
            else:
                gaussians.scales = (self.scale_activation(gaussians.scales) * self.init_scale).clamp_max_(max=self.max_scale)
        else:
            if self.learn_scale:
                gaussians.scales = self.get_scales()
            else:
                assert 0
        
        if self.use_non_rigid_rotations:
            if self.learn_quaternions:
                if self.non_rigid_rotation_mode == 'add':
                    gaussians.quaternions = self.get_quaternions() + gaussians.quaternions
                else:
                    new_quaternions = self.rotation_activation(gaussians.quaternions)
                    gaussians.quaternions = quaternion_multiply(new_quaternions, self.get_quaternions())
            else:
                gaussians.quaternions = self.rotation_activation(gaussians.quaternions)
        else:
            if self.learn_quaternions:
                gaussians.quaternions = self.get_quaternions()
            else:
                assert 0
        
        return gaussians
        
    def animate(self, smpl_observed_inputs: Optional[dict] = None) -> GaussianOutput:

        if smpl_observed_inputs is None:
            smpl_observed_inputs = self.smpl_canonical_inputs
        
        smpl_canonical_inputs = self.smpl_canonical_inputs

        # Get LBS Transform from smpl-canonical to canonical space
        _, canonical_vertex_transform, canonical_transforms = self.lbs_model.forward(**smpl_canonical_inputs)

        # Get LBS Transform from smpl-canonical to observed space
        _, observed_vertex_transform, observed_transforms = self.lbs_model.forward(**smpl_observed_inputs)

        # Get uncontrained gaussians
        positions = self.get_positions()
        lbs_weights = self.get_lbs_weights()
        vertex_indices = self.nearest_triangles_buffer['nearest_vertex_indices']

        # Canonical LBS Transform
        canonical_positions = self.lbs_transform(
            positions=positions,
            transforms=canonical_transforms,
            lbs_weights=lbs_weights,
            vertex_indices=vertex_indices,
        )
        
        # Canonical Forward
        gaussians = self.get_unconstrained_gaussians(
            positions=positions,
            canonical_positions=canonical_positions,
            smpl_observed_inputs=smpl_observed_inputs,
            fix_opacities=False,
        )

        # Observed Non-Rigid Transform
        gaussians = self.non_rigid_transform(gaussians)
        
        # Observed LBS Transform
        gaussians.positions, gaussians.quaternions = self.lbs_transform(
            positions=gaussians.positions,
            transforms=observed_transforms,
            lbs_weights=lbs_weights,
            vertex_indices=vertex_indices,
            quaternions=gaussians.quaternions,
        )

        if self.render_unconstrained_3d_gaussians_only:
            return gaussians
        
        # Get mesh-binding gaussians
        constrained_gaussians_list = []
        if self.learn_betas:
            _, canonical_vertex_transform_with_betas, _ = self.lbs_model.forward(**smpl_canonical_inputs, extra_betas=self._betas)
            _, observed_vertex_transform_with_betas, _ = self.lbs_model.forward(**smpl_observed_inputs, extra_betas=self._betas)

        for body_part, gaussian_model in self.mesh_binding_gaussians.items():
            gaussian_model: MeshBindingGaussianModel
            vertex_coords = gaussian_model.get_vertex_coords()
            if (body_part == 'hands' and self.learn_hand_betas) or \
              (body_part == 'face' and self.learn_face_betas):
                cnl_vertex_transform: RigidTransform = canonical_vertex_transform_with_betas.squeeze(0)
                obs_vertex_transform: RigidTransform = observed_vertex_transform_with_betas.squeeze(0)
            else:
                cnl_vertex_transform: RigidTransform = canonical_vertex_transform.squeeze(0)
                obs_vertex_transform: RigidTransform = observed_vertex_transform.squeeze(0)
            
            # Mesh-independent Forward
            canonical_vertex_coords = cnl_vertex_transform.transform_points(vertex_coords, indices=gaussian_model.predefined_vertex_indices)
            canonical_positions = gaussian_model.get_positions(vertex_coords=canonical_vertex_coords)
            constrained_gaussians = self.get_constrained_gaussians(
                positions=canonical_positions,
                fix_opacities=True,
            )
            # LBS Transform
            observed_vertex_coords = obs_vertex_transform.transform_points(vertex_coords, indices=gaussian_model.predefined_vertex_indices)
            # Mesh-dependent Forward
            constrained_gaussians.positions = gaussian_model.get_positions(vertex_coords=observed_vertex_coords)
            constrained_gaussians.scales, constrained_gaussians.quaternions = gaussian_model.get_scales_and_quaternions(
                vertex_coords=observed_vertex_coords,
                positions=constrained_gaussians.positions,
            )
            # Append
            constrained_gaussians_list.append(constrained_gaussians)
        
        # Merge and Return
        if self.render_mesh_binding_3d_gaussians_only:
            return merge_gaussians(*constrained_gaussians_list)
        
        return merge_gaussians(gaussians, *constrained_gaussians_list)

    def get_optimizer(self, cfg:TrainConfig):
        optimizers = {}
        # Gaussians
        iterations = cfg.optim.iters
        optim_params = OptimizationParams(
            iterations=iterations,
            position_lr_init=cfg.render.position_lr_init,
            position_lr_final=cfg.render.position_lr_final,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=iterations * 2,
            feature_lr=cfg.render.feature_lr,
            opacity_lr=cfg.render.opacity_lr,
            scaling_lr=cfg.render.scaling_lr,
            rotation_lr=cfg.render.rotation_lr,
        )
        gaussian_optimizer = build_optimizer(model=self, cfg=cfg, optim_params=optim_params)
        if gaussian_optimizer is not None:
            optimizers['avatar'] = gaussian_optimizer
        # Deform
        if self.lbs_model is not None:
            lbs_optimizer = None
            params_list = []
            if self.learn_lbs_weights:
                params_list.append({'params': [self._lbs_weights], 'lr': cfg.render.lbs_lr})
            if self.learn_betas:
                params_list.append({'params': [self._betas], 'lr': cfg.render.betas_lr})
            if len(params_list) > 0:
                lbs_optimizer = torch.optim.Adam(params_list, weight_decay=0.0)
            if lbs_optimizer is not None:
                optimizers['lbs'] = lbs_optimizer
        if self.deform_model is not None:
            optimizers['deform'] = self.deform_model.get_optimizer(cfg=cfg)
        # NeRF
        nerf_lr = cfg.nerf.lr
        params = [
            {'params': self.nerf_encoder.parameters(), 'lr': nerf_lr * 10},
            {'params': self.nerf_opacity_and_color_net.parameters(), 'lr': nerf_lr},
            {'params': self.nerf_scale_and_quaternion_net.parameters(), 'lr': nerf_lr},
        ]
        optimizers['nerf'] = torch.optim.Adam(params, betas=(0.9, 0.99), eps=1e-15, weight_decay=0)
        # Mesh-binding Gaussians
        for model_name, gaussian_model in self.mesh_binding_gaussians.items():
            gaussian_model: MeshBindingGaussianModel
            optimizers.update(gaussian_model.get_optimizer(cfg=cfg, optimizer_name='mesh_' + model_name))
        # Return
        return optimizers


Avatar = DreamWaltzG


####################################################################################
def build_gaussian_avatar(cfg: TrainConfig, smpl_prompt: SMPLPrompt, point_cloud=None, nerf=None):
    gs_type = cfg.render.gs_type
    deform_type = cfg.render.deform_type.lower().split(',')
    
    smpl_model = smpl_prompt.smpl_model
    smpl_canonical_inputs = smpl_prompt.smpl_canonical_inputs
    smpl_canonical_outputs = smpl_prompt.smpl_canonical_outputs

    canonical_vertices = smpl_canonical_outputs.vertices[0]  # [10475, 3]
    canonical_triangles = smpl_model.triangles  # [20908, 3]

    kwargs = {
        'cfg': cfg,
        'canonical_vertices': canonical_vertices,
        'canonical_triangles': canonical_triangles,
        'point_cloud': point_cloud,
        'smpl_canonical_inputs': smpl_canonical_inputs,
    }

    if 'lbs' in deform_type:
        kwargs['lbs_model'] = LinearBlendSkinning(
            model=smpl_model.model,
            learn_v_template=cfg.render.deform_learn_v_template,
            learn_shapedirs=cfg.render.deform_learn_shapedirs,
            learn_posedirs=cfg.render.deform_learn_posedirs,
            learn_expr_dirs=cfg.render.deform_learn_expr_dirs,
            learn_lbs_weights=cfg.render.deform_learn_lbs_weights,
            learn_J_regressor=cfg.render.deform_learn_J_regressor,
        )
    if 'glbs' in deform_type:
        kwargs['lbs_model'] = GeneralLinearBlendSkinning(
            model=smpl_model.model,
            learn_v_template=cfg.render.deform_learn_v_template,
            learn_shapedirs=cfg.render.deform_learn_shapedirs,
            learn_posedirs=cfg.render.deform_learn_posedirs,
            learn_expr_dirs=cfg.render.deform_learn_expr_dirs,
            learn_lbs_weights=cfg.render.deform_learn_lbs_weights,
            learn_J_regressor=cfg.render.deform_learn_J_regressor,
        )
    if 'non_rigid' in deform_type:
        kwargs['deform_model'] = DeformNetwork()
    
    if nerf is None and gs_type != 'vanilla':
        nerf = build_NeRFNetwork(cfg=cfg.nerf)

    if gs_type == 'vanilla':
        kwargs['learn_positions'] = cfg.render.learn_positions
        avatar = VanillaAvatar(**kwargs)
    elif gs_type == 'hash':
        avatar = HashAvatar(**kwargs, nerf=nerf)
    elif gs_type == 'hashed_gs_w_mesh':
        body_parts = cfg.predefined_body_parts.split(',')
        predefined_vids, predefined_fids = smpl_model.get_semantic_indices(body_parts)
        kwargs['predefined_vertex_indices'] = predefined_vids
        kwargs['predefined_triangle_indices'] = predefined_fids
        avatar = HashAvatarWithMesh(**kwargs, nerf=nerf)
    elif gs_type == 'dreamwaltz-g':
        body_parts = cfg.predefined_body_parts.split(',')
        predefined_meshes = defaultdict(dict)
        for part_name in body_parts:
            predefined_vids, predefined_fids = smpl_model.get_semantic_indices(part_name)
            predefined_meshes[part_name]['vertex_indices'] = predefined_vids
            predefined_meshes[part_name]['triangle_indices'] = predefined_fids
        kwargs['predefined_meshes'] = predefined_meshes
        kwargs['learn_positions'] = cfg.render.learn_positions
        kwargs['learn_scales'] = cfg.render.learn_scales
        kwargs['learn_quaternions'] = cfg.render.learn_quaternions
        kwargs['learn_lbs_weights'] = cfg.render.learn_lbs_weights
        avatar = DreamWaltzG(**kwargs, nerf=nerf)
    else:
        assert 0, gs_type

    return avatar.to(avatar.device)
