import torch
import numpy as np


def NeRF_data_to_standard(intrinsics, cam2world, H=None, W=None):
    # Tensor to Numpy
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(cam2world, torch.Tensor):
        cam2world = cam2world.cpu().numpy()
    # Intrinsics
    intrinsics = intrinsics_Vec2Mat(intrinsics, H=H, W=W)
    # Extrinsic
    if cam2world.ndim == 3:
        assert cam2world.shape[0] == 1
        cam2world = cam2world[0]
    extrinsic = SE3_inverse(cam2world)
    return intrinsics, extrinsic


def intrinsics_Vec2Mat(intrinsics, H=None, W=None):
    """
    Input:
        intrinsics: np.array, shape = (4,)
    Return:
        intrinsics: np.array, shape = (3, 3)
    """
    assert intrinsics.ndim == 1 and intrinsics.shape[-1] == 4
    if H is None or W is None:
        K = np.array([
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ])
    else:
        Hc, Wc = intrinsics[2] * 2, intrinsics[3] * 2
        K = np.array([
            [intrinsics[0] * H / Hc, 0.0, H / 2],
            [0.0, intrinsics[1] * W / Wc, W / 2],
            [0.0, 0.0, 1.0],
        ])
    return K


def SE3_Mat2RT(extrinsic):
    """
    Input:
        extrinsic: np.array, shape = (N, 4, 4) or (4, 4)
    Return:
        R: np.array, shape = (N, 3, 3) or (3, 3)
        T: np.array, shape = (N, 3, 1) or (3, 1)
    """
    if extrinsic.ndim == 2:
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3][:, np.newaxis]
    elif extrinsic.ndim == 3:
        R = extrinsic[:, :3, :3]
        T = extrinsic[:, :3, 3][:, np.newaxis]
    return R, T


def SE3_RT2Mat(R, T):
    """
    Input:
        R: np.array, shape = (N, 3, 3) or (3, 3)
        T: np.array, shape = (N, 3, 1) or (3, 1)
    Return:
        extrinsic: np.array, shape = (N, 4, 4) or (4, 4)
    """
    if R.ndim == 2 and T.ndim == 2:
        extrinsic = np.zeros((4, 4))
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T[:, 0]
    elif R.ndim == 3 and T.ndim == 3:
        extrinsic = np.zeros((R.shape[0], 4, 4))
        extrinsic[:, :3, :3] = R
        extrinsic[:, :3, 3] = T[:, :, 0]
    return extrinsic


def SE3_inverse(mat):
    """
    Input:
        mat: np.array, shape = (4, 4)
    Return:
        mat: np.array, shape = (4, 4)
    """
    R, T = SE3_Mat2RT(mat)
    R = np.linalg.inv(R)
    T = np.dot(R, - T)
    return SE3_RT2Mat(R, T)
