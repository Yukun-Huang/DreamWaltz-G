import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion
from typing import Tuple, Mapping, Any, Optional
from loguru import logger

from .gaussian_utils import GaussianOutput, inverse_sigmoid
from .spherical_harmonics import RGB2SH


class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device: torch.device = None

        self._n_points = None  # torch.empty(0)
        self._positions = None  # torch.empty(0)
        self._sh_features_dc = None  # torch.empty(0)
        self._sh_features_rest = None  # torch.empty(0)
        self._scales = None  # torch.empty(0)
        self._quaternions = None  # torch.empty(0)
        self._opacities = None  # torch.empty(0)
            
        self.scale_activation = torch.exp
        self.scale_inverse_activation = torch.log

        self.color_activation = torch.sigmoid

        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def get_positions(self):
        return self._positions
    
    def get_sh_features(self):
        features_dc = self._sh_features_dc
        features_rest = self._sh_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_opacities(self, return_ones=False):
        if not return_ones:
            return self.opacity_activation(self._opacities.view(-1, 1))
        else:
            return torch.ones_like(self._opacities.view(-1, 1))
    
    def get_scales(self, return_means=False):
        if not return_means:
            return self.scale_activation(self._scales)
        else:
            return self.scale_activation(self._scales.mean(dim=-1, keepdim=True).expand(-1, 3))
    
    def get_quaternions(self):
        return self.rotation_activation(self._quaternions)

    def reset_by_state_dict(
        self,
        state_dict: Mapping[str, Any],
        attribute_names: Tuple[str] = ('_positions', '_sh_features_dc', '_sh_features_rest', '_scales', '_quaternions', '_opacities'),
    ):
        def _reset(name:str, n_points:int):
            if not hasattr(self, name):
                return
            params_tensor = getattr(self, name).data
            requires_grad = getattr(self, name).requires_grad
            new_params_tensor = torch.empty(
                n_points, *params_tensor.shape[1:],
                dtype=params_tensor.dtype,
                device=params_tensor.device,
            )
            setattr(self, name, nn.Parameter(new_params_tensor, requires_grad=requires_grad))

        for k, v in state_dict.items():
            if v.ndim == 0:
                continue
            if k not in attribute_names:
                continue
            if len(getattr(self, k)) != len(v):
                _reset(name=k, n_points=len(v))
        
        self._n_points = state_dict['_positions'].shape[0]

        logger.info(f"Reset Gaussian model with {self._n_points} Gaussians.")

    def forward(self) -> GaussianOutput:
        return GaussianOutput(
            positions=self.get_positions(),
            sh_features=self.get_sh_features(),
            opacities=self.get_opacities(),
            quaternions=self.get_quaternions(),
            scales=self.get_scales(),
        )

    @torch.no_grad()
    def load_ply(
        self,
        ply_path: str,
        max_sh_degree: int = 3,
        device: str = "cuda",
    ):
        from plyfile import PlyData, PlyElement
        plydata = PlyData.read(ply_path)

        xyz = np.stack((
            + np.asarray(plydata.elements[0]["x"]),
            + np.asarray(plydata.elements[0]["y"]),
            + np.asarray(plydata.elements[0]["z"]),
        ),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return {
            'positions': torch.tensor(xyz, dtype=torch.float, device=device),
            'sh_features_dc': torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous(),
            'sh_features_rest': torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous(),
            'opacities': torch.tensor(opacities, dtype=torch.float, device=device),
            'scales': torch.tensor(scales, dtype=torch.float, device=device),
            'quaternions': torch.tensor(rots, dtype=torch.float, device=device),
        }

    @torch.no_grad()
    def load_ply_and_initialize(
        self,
        path: str,
        max_sh_degree: int = 3,
    ):
        data = self.load_ply(path, max_sh_degree=max_sh_degree)
        
        positions: torch.Tensor = data['positions']
        sh_features_dc: torch.Tensor = data['sh_features_dc']
        sh_features_rest: torch.Tensor = data['sh_features_rest']
        opacities: torch.Tensor = data['opacities']
        scales: torch.Tensor = data['scales']
        quaternions: torch.Tensor = data['quaternions']

        self._positions = nn.Parameter(positions.requires_grad_(True))
        self._sh_features_dc = nn.Parameter(sh_features_dc.requires_grad_(True))
        self._sh_features_rest = nn.Parameter(sh_features_rest.requires_grad_(True))
        self._opacities = nn.Parameter(opacities.requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self._quaternions = nn.Parameter(quaternions.requires_grad_(True))

        self._n_points = positions.shape[0]

    def __repr__(self):
        return f"3D Gaussian model \"{self.__class__.__name__}\" with {self._n_points} Gaussians."


class SuGaRModel(GaussianModel):
    def __init__(
        self,
        device: torch.device,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        n_gaussians_per_triangle: int = 6,
        sh_levels: int = 4,
        surface_mesh_thickness: float = 0.0,
        init_opacity: float = 0.99,
        learn_positions: bool = False,
        learn_opacities: bool = True,
        learn_colors: bool = True,
        learn_scales: bool = True,
        learn_quaternions: bool = True,
    ) -> None:
        """
        Args:
            vertices: torch.Tensor, [V, 3]
            faces: torch.Tensor, [F, 3]
        """
        super().__init__()
        self.device = device
        self.binded_to_surface_mesh = True

        self.n_gaussians_per_triangle = n_gaussians_per_triangle
        self.surface_triangle_circle_radius, self.surface_triangle_bary_coords = self.prepare_mesh_binding_params()
        self.surface_mesh_thickness = torch.nn.Parameter(torch.tensor(surface_mesh_thickness), requires_grad=False)

        n_vertices = vertices.shape[0]
        n_faces = faces.shape[0]
        n_points = n_faces * n_gaussians_per_triangle
        self._n_points = n_points

        # Initialize positions
        self._offsets = nn.Parameter(torch.zeros(n_vertices, 3), requires_grad=learn_positions)

        # Initialize SH features
        self.sh_levels = sh_levels
        colors = torch.rand(n_points, 3)
        self._sh_features_dc = nn.Parameter(RGB2SH(colors).unsqueeze(dim=1), requires_grad=learn_colors)
        self._sh_features_rest = nn.Parameter(torch.zeros(n_points, sh_levels**2 - 1, 3), requires_grad=learn_colors)

        # Initialize opacities
        opacities = self.opacity_inverse_activation(init_opacity * torch.ones((n_points, 1)))
        self._opacities = nn.Parameter(opacities, requires_grad=learn_opacities)

        # Initialize scales
        faces_verts = vertices[faces]  # [F, 3=vid, 3=xyz]

        edges = (faces_verts - faces_verts[:, [1, 2, 0], :]).norm(dim=-1).min(dim=-1).values  # [F,]
        scales = (edges * self.surface_triangle_circle_radius * 2).clamp_min(1e-7)  # [F,]
        scales = scales.reshape(-1, 1, 1).repeat(1, n_gaussians_per_triangle, 2).reshape(-1, 2)  # [N, 2]
        self._scales = nn.Parameter(self.scale_inverse_activation(scales), requires_grad=learn_scales)

        # Initialize quaternions
        # We actually don't learn quaternions here, but complex numbers to encode a 2D rotation in the triangle's plane
        complex_numbers = torch.zeros(n_points, 2)
        complex_numbers[:, 0] = 1.
        self._quaternions = nn.Parameter(complex_numbers, requires_grad=learn_quaternions)

        self.to(device)

    def prepare_mesh_binding_params(self) -> Tuple[float, torch.Tensor]:
        device = self.device
        n_gaussians_per_triangle = self.n_gaussians_per_triangle

        if n_gaussians_per_triangle == 1:
            surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
            surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3]],
                dtype=torch.float32,
                device=device,
            )[..., None]

        if n_gaussians_per_triangle == 3:
            surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
            surface_triangle_bary_coords = torch.tensor(
                [[1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]],
                dtype=torch.float32,
                device=device,
            )[..., None]

        if n_gaussians_per_triangle == 4:
            surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
            surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3],
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3]],
                dtype=torch.float32,
                device=device,
            )[..., None]  # n_gaussians_per_face, 3, 1

        if n_gaussians_per_triangle == 6:
            surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
            surface_triangle_bary_coords = torch.tensor(
                [[2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
                [1/6, 5/12, 5/12],
                [5/12, 1/6, 5/12],
                [5/12, 5/12, 1/6]],
                dtype=torch.float32,
                device=device,
            )[..., None]
        
        return surface_triangle_circle_radius, surface_triangle_bary_coords

    def get_scales(self):
        scales = torch.cat([
            self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
            self.scale_activation(self._scales),
            ], dim=-1)
        return scales

    def get_quaternions(self, mesh):

        # We compute quaternions to enforce face normals to be the first axis of gaussians
        R_0 = torch.nn.functional.normalize(mesh.fn, dim=-1)

        # We use the first side of every triangle as the second base axis
        faces_verts = mesh.v[mesh.f]
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)

        # We use the cross product for the last base axis
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        
        # We now apply the learned 2D rotation to the base quaternion
        complex_numbers = torch.nn.functional.normalize(self._quaternions, dim=-1).view(len(mesh.f), self.n_gaussians_per_triangle, 2)
        R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
        R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]

        # We concatenate the three vectors to get the rotation matrix
        R = torch.cat([R_0[:, None, ..., None].expand(-1, self.n_gaussians_per_triangle, -1, -1).clone(),
                    R_1[..., None],
                    R_2[..., None]],
                    dim=-1).view(-1, 3, 3)
        quaternions = matrix_to_quaternion(R)

        return torch.nn.functional.normalize(quaternions, dim=-1)
    
    def get_positions(self, mesh):

        # First gather vertices of all triangles
        faces_verts = (mesh.v)[mesh.f]  # [F, n_vertices_per_face=3, n_coords=3]
        
        # Then compute the points using barycenter features in the surface triangles
        points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # [F, n_gaussians_per_face, 3, n_coords=3]
        points = points.sum(dim=-2)  # [F, n_gaussians_per_face, n_coords=3]
        
        return points.reshape(-1, 3)  # [F * n_gaussians_per_face, n_coords=3]


class GaMeSModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self._scales = torch.empty(0)
        self._flame_shape = torch.empty(0)
        self._flame_exp = torch.empty(0)
        self._flame_pose = torch.empty(0)
        self._flame_neck_pose = torch.empty(0)
        self._flame_trans = torch.empty(0)
        self.faces = torch.empty(0)
        self._vertices_enlargement = torch.empty(0)

    @property
    def get_xyz(self):
        return self._xyz

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _xyz = torch.matmul(
            self.alpha,
            self.vertices[self.faces]
        )
        self._xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )

    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from centroid
        to 2nd vertex onto subspace spanned by v0 and v1
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

        triangles = self.vertices[self.faces]
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = eps * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))

        self._scaling = torch.log(
            torch.nn.functional.relu(self._scales * scales.flatten(start_dim=0, end_dim=1)) + eps
        )

        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0

        """
        self.alpha = self.update_alpha_func(self._alpha)
        vertices, _ = self.point_cloud.flame_model(
            shape_params=self._flame_shape,
            expression_params=self._flame_exp,
            pose_params=self._flame_pose,
            neck_pose=self._flame_neck_pose,
            transl=self._flame_trans
        )
        self.vertices = self.point_cloud.transform_vertices_function(
            vertices,
            self._vertices_enlargement
        )
        self._calc_xyz()
