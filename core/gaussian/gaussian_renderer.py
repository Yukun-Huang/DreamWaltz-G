import torch
from pytorch3d.transforms import quaternion_to_matrix
from typing import Optional

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .gaussian_utils import get_colors, GaussianOutput


class GaussianRenderer:
    def __init__(
        self,
        sh_levels=4,
        bg_color=(0.0, 0.0, 0.0),
        compute_color_in_rasterizer=True,
        compute_covariance_in_rasterizer=True,
    ) -> None:

        self.sh_levels = sh_levels
        self.bg_color = torch.tensor(bg_color)
        self.compute_color_in_rasterizer = compute_color_in_rasterizer
        self.compute_covariance_in_rasterizer = compute_covariance_in_rasterizer

    def build_gaussian_rasterizer(self, data: dict, **kwargs) -> GaussianRasterizer:
        world_view_matrix = data['extrinsic'][0]
        projection_matrix = data['projection'][0]
        image_width = data['image_width']
        image_height = data['image_height']
        tanfovy = data['tanfov'][0].item()
        if 'tanfov_x' in data:
            tanfovx = data['tanfov_x'][0].item()
        else:
            tanfovx = tanfovy
        
        device = world_view_matrix.device

        viewmatrix = world_view_matrix.transpose(0, 1)
        projmatrix = viewmatrix @ projection_matrix.transpose(0, 1)
        bg_color = self.bg_color.to(device)

        campos = data['c2w'][0, :3, 3]
        sh_degree = self.sh_levels - 1

        raster_settings = {
            "image_height": image_height,
            "image_width": image_width,
            "tanfovx": tanfovx,
            "tanfovy": tanfovy,
            "bg": bg_color,
            "viewmatrix": viewmatrix,
            "projmatrix": projmatrix,
            "sh_degree": sh_degree,
            "campos": campos,
        }
        raster_settings.update(kwargs)
        
        for key, value in raster_settings.items():
            if isinstance(value, torch.Tensor):
                raster_settings[key] = value.to(device)

        raster_settings = GaussianRasterizationSettings(
            **raster_settings,
            scale_modifier=1.,
            prefiltered=False,
            debug=False,
        #     kernel_size=kernel_size,
        #     subpixel_offset=subpixel_offset,
        #     scale_modifier=scaling_modifier,
        )

        return GaussianRasterizer(raster_settings=raster_settings)

    def compute_colors(
        self,
        sh_features: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        camera_positions: Optional[torch.Tensor] = None,
        sh_levels: Optional[int] = None,
        sh_rotations: Optional[torch.Tensor] = None,
    ):
        """Returns the RGB color of the points for the given camera pose.

        Args:
            positions (torch.Tensor, optional): Shape (n_pts, 3). Defaults to None.
            camera_positions (torch.Tensor, optional): Shape (n_pts, 3) or (1, 3). Defaults to None.
            directions (torch.Tensor, optional): Shape (n_pts, 3) or (1, 3). Defaults to None.
            sh_rotations (torch.Tensor, optional): Shape (n_pts, 3) or (1, 3). Defaults to None.

        Returns:
            colors: (torch.Tensor): Shape (n_pts, 3).

        """
        if directions is None:
            if positions is None or camera_positions is None:
                raise ValueError("Either directions or positions must be provided.")

            directions = torch.nn.functional.normalize(positions - camera_positions, dim=-1)

        if sh_rotations is not None:
            directions = (directions.unsqueeze(1) @ sh_rotations)[..., 0, :]
        
        if sh_levels is None:
            sh_levels = self.sh_levels

        return get_colors(sh_features=sh_features, directions=directions, sh_levels=sh_levels)
    
    @staticmethod
    def compute_3d_covariance(scales, quaternions):
        device = scales.device

        cov3Dmatrix = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device=device)
        rotation = quaternion_to_matrix(quaternions)

        cov3Dmatrix[:,0,0] = scales[:,0]**2
        cov3Dmatrix[:,1,1] = scales[:,1]**2
        cov3Dmatrix[:,2,2] = scales[:,2]**2
        cov3Dmatrix = rotation @ cov3Dmatrix @ rotation.transpose(-1, -2)
        
        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=device)

        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
        
        return cov3D

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def render(
        self,
        data: dict,
        gaussians: GaussianOutput,
        return_2d_radii: bool = False,
        rasterizer: Optional[GaussianRasterizer] = None,
    ) -> dict:
        """Render an image using the Gaussian Splatting Rasterizer.

        Args:
            data (CamerasWrapper): _description_.
            return_2d_radii (bool, optional): _description_. Defaults to False.

        Returns:
            image: torch.Tensor, [B, H, W, 3], 0.0 ~ 1.0
            depth: torch.Tensor, [B, H, W, 1], 0.0 ~ +inf
            alpha: torch.Tensor, [B, H, W, 1], 0.0 ~ 1.0
            screenspace_points: torch.Tensor, [N, 3]
            splat_opacities: torch.Tensor, [N, 1]
            splat_colors: torch.Tensor, [N, 3]
            radii: torch.Tensor, [N,]
        """
        if rasterizer is None:
            rasterizer = self.build_gaussian_rasterizer(data=data)

        if gaussians.colors is not None:
            gaussians.sh_features = None
        elif not self.compute_color_in_rasterizer:
            camera_positions = data['c2w'][:, :3, 3]
            gaussians.colors = self.compute_colors(
                sh_features=gaussians.sh_features,
                positions=gaussians.positions,
                camera_positions=camera_positions,
            )
            gaussians.sh_features = None

        if not self.compute_covariance_in_rasterizer:
            gaussians.cov3D = self.compute_3d_covariance(
                scales=gaussians.scales,
                quaternions=gaussians.quaternions,
            )
            gaussians.quaternions = None
            gaussians.scales = None
        
        means3D = gaussians.positions

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros(means3D.shape[0], 3, dtype=means3D.dtype, requires_grad=True, device=means3D.device)
        if return_2d_radii:
            try:
                screenspace_points.retain_grad()
            except:
                print("WARNING: return_2d_radii is True, but failed to retain grad of screenspace_points!")
        means2D = screenspace_points
        
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = gaussians.sh_features,
            colors_precomp = gaussians.colors,
            opacities = gaussians.opacities,
            scales = gaussians.scales,
            rotations = gaussians.quaternions,
            cov3D_precomp = gaussians.cov3D,
        )

        # rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        #     means3D = means3D,
        #     means2D = means2D,
        #     shs = gaussians.sh_features,
        #     colors_precomp = gaussians.colors,
        #     opacities = gaussians.opacities,
        #     scales = gaussians.scales,
        #     rotations = gaussians.quaternions,
        #     cov3D_precomp = gaussians.cov3D,
        # )

        # rendered_depth = rendered_depth / rendered_alpha
        # rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)

        # rendered_normal = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
        # rendered_normal = rendered_normal * 0.5 + 0.5

        outputs = {
            "image": rendered_image.permute(1, 2, 0).unsqueeze(0),  # [3, H, W] -> [B, H, W, 3]
            "depth": rendered_depth.permute(1, 2, 0).unsqueeze(0),  # [1, H, W] -> [B, H, W, 1]
            "alpha": rendered_alpha.permute(1, 2, 0).unsqueeze(0),  # [1, H, W] -> [B, H, W, 1]
        }

        if return_2d_radii:
            outputs["radii"] = radii
            outputs["viewspace_points"] = screenspace_points
        
        return outputs
