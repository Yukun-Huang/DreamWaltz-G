import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from PIL import Image


def to_trimesh(
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor = None,
    vertex_colors: np.ndarray | torch.Tensor = None,
    **kwargs,
):
    """
        vertices: [V, 3], np.ndarray
        faces: [F, 3], np.ndarray
        vertex_colors: [V, 3], np.ndarray
    """
    assert vertices.ndim == 2

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    
    if isinstance(vertex_colors, torch.Tensor):
        vertex_colors = vertex_colors.detach().cpu().numpy()
    elif isinstance(vertex_colors, str):
        COLOR_MAP = {
            'gray': [0.3, 0.3, 0.3, 0.8],
            'red': [1.0, 0.0, 0.0, 0.8],
            'green': [0.0, 1.0, 0.0, 0.8],
            'blue': [0.0, 0.0, 1.0, 0.8],
            'white': [1.0, 1.0, 1.0, 0.8],
            'black': [0.0, 0.0, 0.0, 0.8],
            'yellow': [1.0, 1.0, 0.0, 0.8],
            'purple': [1.0, 0.0, 1.0, 0.8],
            'cyan': [0.0, 1.0, 1.0, 0.8],
        }
        vertex_colors = np.ones([vertices.shape[0], 4]) * COLOR_MAP[vertex_colors]

    if faces is None:
        return trimesh.points.PointCloud(vertices, colors=vertex_colors)
    else:
        return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, **kwargs)


def sample_points_from_trimesh(mesh: trimesh.Trimesh, N: int, sample_color=False):
    samples, _ = trimesh.sample.sample_surface(mesh, N, sample_color=sample_color)
    return samples


def render_trimesh(mesh: trimesh.Trimesh, extrinsic: np.ndarray, intrinsics: np.ndarray, width: int, height: int, fix_z_axis:bool=True) -> Image.Image:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import pyrender

    # generate mesh
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    # camera params
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    xfov = np.arctan(abs(cx / fx)) * 2
    yfov = np.arctan(abs(cy / fy)) * 2

    # compose scene
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
    camera = pyrender.PerspectiveCamera(yfov=yfov)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    # camera pose
    camera_pose = np.linalg.inv(extrinsic)
    if fix_z_axis:
        camera_pose[:3, 2] *= -1
    camera_pose[3, :] = np.array([0., 0., 0., 1.])

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=camera_pose)
    scene.add(camera, pose=camera_pose)

    # render scene
    r = pyrender.OffscreenRenderer(width, height)
    color, depth = r.render(scene)
    r.delete()

    return Image.fromarray(color, mode='RGB')
