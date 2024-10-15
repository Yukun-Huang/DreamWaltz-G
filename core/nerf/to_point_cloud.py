import numpy as np
import torch
from loguru import logger
from typing import Iterable

from .nerf_utils import custom_meshgrid
from utils.point_cloud import BasicPointCloud


def latent_to_rgb(albedos):
    assert albedos.ndim == 2 and albedos.size(1) in (3, 4)
    if albedos.shape[1] == 3:
        rgbs = albedos
    else:
        assert albedos.shape[1] == 4
        decode_mat = torch.tensor([
            #   R       G       B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], device=albedos.device)  # [4, 3]
        rgbs = albedos.matmul(decode_mat)
    return rgbs


@torch.inference_mode()
def export_point_cloud(self, resolution=None, split_size=128, density_thresh=None) -> BasicPointCloud:

    logger.info(f'Extracting point cloud from NeRF...')

    self.update_extra_state()

    if resolution is None:
        resolution = self.grid_size  # self.grid_size: 128

    if density_thresh is None:
        if self.cuda_ray:
            density_thresh = min(self.mean_density, self.density_thresh) \
                if np.greater(self.mean_density, 0) else self.density_thresh
        else:
            density_thresh = self.density_thresh

    # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
    if self.density_activation == 'softplus':
        density_thresh = density_thresh * 25

    # # query
    X = torch.linspace(-1, 1, resolution).split(split_size)
    Y = torch.linspace(-1, 1, resolution).split(split_size)
    Z = torch.linspace(-1, 1, resolution).split(split_size)

    pc = BasicPointCloud()
    min_density = self.max_density
    max_density = 0.0

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = custom_meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                pts = pts.to(self.aabb_train.device)

                sigmas, albedos = self.common_forward(pts)
                sigmas = sigmas.reshape(-1, 1)
                albedos = latent_to_rgb(albedos)
                normals = self.normal(pts)

                min_density = min(min_density, sigmas.min().item())
                max_density = max(max_density, sigmas.max().item())

                valid_indices = (sigmas > density_thresh).flatten()

                pts = pts[valid_indices].cpu().numpy()
                albedos = albedos[valid_indices].cpu().numpy()
                sigmas = sigmas[valid_indices].cpu().numpy()
                normals = normals[valid_indices].cpu().numpy()

                pc.points = np.concatenate((pc.points, pts), axis=0)
                pc.colors = np.concatenate((pc.colors, albedos), axis=0)
                pc.alphas = np.concatenate((pc.alphas, sigmas), axis=0)
                pc.normals = np.concatenate((pc.normals, normals), axis=0)

    logger.info(f'Extracting point cloud done! Obtain {pc.points.shape[0]} points!')
    logger.info(f'    density thresh: {density_thresh} ({min_density} ~ {max_density})')

    # valid_indices = (pc.alphas > density_thresh).flatten()
    # pc.points = pc.points[valid_indices]
    # pc.colors = pc.colors[valid_indices]
    # pc.alphas = pc.alphas[valid_indices]
    # pc.normals = pc.normals[valid_indices]
    return pc


def remove_points_inside_bboxes(point_cloud: BasicPointCloud, bboxes: Iterable[Iterable[Iterable[float]]]) -> BasicPointCloud:
    mask = np.full(len(point_cloud), True, dtype=bool)

    if isinstance(bboxes[0][0], float):
        bboxes = [bboxes,]

    for i, point in enumerate(point_cloud.points):
        for bbox in bboxes:
            min_corner = np.amin(bbox, axis=0)
            max_corner = np.amax(bbox, axis=0)
            if np.all(point >= min_corner) and np.all(point <= max_corner):
                mask[i] = False
                break

    point_cloud.points = point_cloud.points[mask]
    point_cloud.colors = point_cloud.colors[mask]
    point_cloud.normals = point_cloud.normals[mask]
    point_cloud.alphas = point_cloud.alphas[mask]

    return point_cloud
