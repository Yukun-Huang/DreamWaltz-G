import torch
import torch.utils.dlpack
import numpy as np
import open3d as o3d
from typing import List


def build_ray_casting_scene(vertices: np.ndarray, triangles: np.ndarray):
    ray_casting_scene = o3d.t.geometry.RaycastingScene()
    for each_vertices in vertices:
        mesh = o3d.geometry.TriangleMesh(
            vertices = o3d.utility.Vector3dVector(each_vertices),
            triangles = o3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        ray_casting_scene.add_triangles(mesh_t)
    return ray_casting_scene


def cast_rays(
        ray_casting_scene,
        return_type: str,
        extrinsic: np.ndarray,
        intrinsics: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
    """
    Input:
        intrinsics: np.array, [3, 3]
        extrinsic: np.array, [4, 4], world -> camera
    """
    # Rays are 6D vectors with origin and ray direction.
    # Here we use a helper function to create rays for a pinhole camera.
    rays = ray_casting_scene.create_rays_pinhole(intrinsics, extrinsic, width_px=width, height_px=height)

    # Compute the ray intersections.
    ans = ray_casting_scene.cast_rays(rays)

    if return_type == 'depth':
        return ans['t_hit'].numpy()
    
    if return_type == 'normal':
        return ans['primitive_normals'].numpy()


def export_distance_from_ray_casting_scene(ray_casting_scene, query_points: torch.Tensor, signed=True):
    """
    Input:
        query_points: torch.Tensor, [..., 3]
    Return:
        distances: torch.Tensor, [...]
    """
    if isinstance(query_points, torch.Tensor):
        device = query_points.device
        query_points = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query_points.detach().cpu()))
    if signed:
        distances = ray_casting_scene.compute_signed_distance(query_points)
    else:
        distances = ray_casting_scene.compute_distance(query_points)
    distances = torch.utils.dlpack.from_dlpack(distances.to_dlpack()).to(device)
    return distances


def signed_distance_to_density(distance, a=0.001):

    def inv_softplus(bias):
        """Inverse softplus function.
        Args:
            bias (float or tensor): the value to be softplus-inverted.
        """
        is_tensor = True
        if not isinstance(bias, torch.Tensor):
            is_tensor = False
            bias = torch.tensor(bias)
        out = bias.expm1().clamp_min(1e-6).log()
        if not is_tensor and out.numel() == 1:
            return out.item()
        return out

    # print(torch.min(distances), torch.mean(distances), torch.max(distances))
    density = torch.sigmoid(- distance / a) / a  # [0, 1000]
    # t = torch.sigmoid(- distances / a) / a  # [0, 1000]
    # density = torch.clamp(inv_softplus(t), min=0.0, max=1/a)

    return density


def build_geometry(vertices, triangles, points=None):
    geometry = []
    for each_vertices in vertices:
        mesh = o3d.geometry.TriangleMesh(
            vertices = o3d.utility.Vector3dVector(each_vertices),
            triangles = o3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        geometry.append(mesh)
    if points is not None:
        points = points.reshape(-1, 3)
        points_pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        points_pcl.paint_uniform_color([1.0, 0.0, 0.0])
        geometry.append(points_pcl)
    # import open3d.web_visualizer
    # open3d.web_visualizer.draw(geometry)
    return geometry


def export_mesh_to_file(vertices, triangles, filename) -> None:
    geometry = build_geometry(vertices, triangles)
    o3d.io.write_triangle_mesh(str(filename), geometry[0], write_triangle_uvs=False)
