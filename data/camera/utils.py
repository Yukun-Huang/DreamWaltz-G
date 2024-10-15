from random import choice, random
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d.renderer.cameras import look_at_view_transform, _get_sfm_calibration_matrix
from loguru import logger
from typing import Optional, Tuple, List, Iterable
from jaxtyping import Float
from numbers import Number

from configs import DataConfig
from core.human.smpl_utils import OPENPOSE_KEYPOINT_NAMES


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def get_tan_half_fov(fov: Tensor, degrees:bool=True):
    if degrees:
        fov = fov * torch.pi / 180.0
    return torch.tan(fov / 2)


def RT_to_SE3(R: Tensor, T: Tensor) -> Tensor:
    """
    Input:
        R: Tensor, (N, 3, 3) or (3, 3)
        T: Tensor, (N, 3, 1) or (3, 1)
    Return:
        M: Tensor, (N, 4, 4) or (4, 4)
    """
    E = torch.eye(4, dtype=R.dtype, device=R.device)
    if R.ndim == 3:
        E = E.view(1, 4, 4).repeat(R.shape[0], 1, 1)
    E[..., :3, :3] = R
    E[..., :3, 3:] = T
    return E


def SE3_to_RT(E: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Input:
        M: Tensor, (N, 4, 4) or (4, 4)
    Return:
        R: Tensor, (N, 3, 3) or (3, 3)
        T: Tensor, (N, 3, 1) or (3, 1)
    """
    R = E[..., :3, :3]
    T = E[..., :3, 3:]
    return R, T


def SE3_inverse(E: Tensor) -> Tensor:
    R, T = SE3_to_RT(E)
    R_inv = torch.inverse(R)
    T_inv = R_inv @ (- T)
    return RT_to_SE3(R_inv, T_inv)


def angle2sphere(
    radius: Tensor,
    elevation: Tensor,
    azimuth: Tensor,
    degrees: bool = True,
) -> Tensor:
    if degrees:
        azimuth = azimuth * torch.pi / 180.0
        elevation = elevation * torch.pi / 180.0
    positions = torch.stack([
        radius * torch.sin(elevation) * torch.sin(azimuth),
        radius * torch.cos(elevation),
        radius * torch.sin(elevation) * torch.cos(azimuth),
    ], dim=-1)
    return positions  # [B, 3]


def to_extrinsic(
    radius: Tensor,
    azimuth: Tensor,
    elevation: Tensor,
    at_vector = ((0, 0, 0),),
    up_vector = ((0, 1, 0),),
) -> Tensor:
    """
    Args:
        radius: torch.Tensor, [B,]
        azimuth: torch.Tensor, [B,]
        elevation: torch.Tensor, [B,]
        at_vector: torch.Tensor or np.ndarray or tuple, [B, 3]
        up_vector: torch.Tensor or np.ndarray or tuple, [B, 3]
    """
    batch_size = radius.shape[0]
    device = radius.device

    if not isinstance(up_vector, Tensor):
        up_vector = torch.tensor(up_vector, dtype=torch.float, device=device).repeat(batch_size, 1)
    if not isinstance(at_vector, Tensor):
        at_vector = torch.tensor(at_vector, dtype=torch.float, device=device).repeat(batch_size, 1)
    spherical_camera_position = angle2sphere(radius=radius, azimuth=azimuth, elevation=elevation)
    camera_position = at_vector + spherical_camera_position
    lookat_vector = safe_normalize(- spherical_camera_position)
    right_vector = safe_normalize(torch.cross(lookat_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, lookat_vector, dim=-1))

    c2w = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 4, 4]

    c2w[:, :3, :3] = torch.stack((right_vector, up_vector, lookat_vector), dim=-1)
    c2w[:, :3, 3] = camera_position

    extrinsic = torch.inverse(c2w)
    return extrinsic, c2w


def to_intrinsics(
    tanfov: Tensor,
    image_height: int,
    image_width: int,
) -> Tensor:
    """
    Args:
        tanfov: torch.Tensor, [B,]
    Return:
        intrinsics: torch.Tensor, [B, 3, 3]
    """
    N, device = tanfov.shape[0], tanfov.device

    focal_length = image_height / (2 * tanfov)
    fx = fy = focal_length

    principal_point = torch.tensor([image_height//2, image_width//2], device=device).repeat(N, 1)
    px, py = principal_point.unbind(1)

    K = torch.zeros(N, 3, 3, device=device)

    x_sign = 1
    y_sign = -1

    K[:, 0, 0] = fx * x_sign
    K[:, 0, 2] = px
    K[:, 1, 1] = fy * y_sign
    K[:, 1, 2] = py
    K[:, 2, 2] = 1

    return K


def to_projection(tanfov: Tensor, z_near: float, z_far: float, aspect_wh:float=1.0, z_range=(-1, 1), tanfov_x: Optional[Tensor] = None) -> Tensor:
    """
    An example of projection matrix:
        [[[ 1.9210,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -1.9210,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  1.0000, -0.0200],
          [ 0.0000,  0.0000,  1.0000,  0.0000]]]
    """
    N, device = tanfov.shape[0], tanfov.device
    
    max_y = tanfov * z_near
    min_y = -max_y
    if tanfov_x is None:
        max_x = max_y * aspect_wh
    else:
        max_x = tanfov_x * z_near
    min_x = -max_x

    x_sign = 1.0
    y_sign = - 1.0  # add a negative sign here as the y axis is flipped in nvdiffrast output

    # NOTE: In OpenGL the projection matrix changes the handedness of the
    # coordinate frame. i.e the NDC space positive z direction is the
    # camera space negative z direction. This is because the sign of the z
    # in the projection matrix is set to -1.0.
    # In pytorch3d we maintain a right handed coordinate system throughout
    # so the so the z sign is 1.0.
    z_sign = 1.0

    K = torch.zeros((N, 4, 4), dtype=torch.float32, device=device)

    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    K[:, 0, 0] = x_sign * 2.0 * z_near / (max_x - min_x)
    K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)

    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    K[:, 1, 1] = y_sign * 2.0 * z_near / (max_y - min_y)
    K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)

    if z_range == (0, 1):
        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * z_far / (z_far - z_near)
        K[:, 2, 3] = -(z_far * z_near) / (z_far - z_near)
    else:
        # NOTE: This maps the z coordinate from [-1, +1].
        K[:, 2, 2] = z_sign * (z_far + z_near) / (z_far - z_near)
        K[:, 2, 3] = -(2 * z_far * z_near) / (z_far - z_near)

    K[:, 3, 2] = z_sign

    return K


def to_screen(mvp: Tensor, image_height: int, image_width: int, with_xyflip:bool=False):
    """
    Return:
        K: torch.Tensor, [B, 4, 4]
    """
    K = torch.zeros((mvp.shape[0], 4, 4), device=mvp.device, dtype=torch.float32)

    xy_sign = -1 if with_xyflip else 1

    K[:, 0, 0] = xy_sign * ((image_width - 1.0) / 2.0)
    K[:, 1, 1] = xy_sign * ((image_height - 1.0) / 2.0)
    K[:, 0, 3] = (image_width - 1.0) / 2.0
    K[:, 1, 3] = (image_height - 1.0) / 2.0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0

    return K


def depth_to_ndc_depth(depth: Tensor, z_near: float, z_far: float, z_range=(-1, 1)):
    if z_range == (-1, 1):
        return (z_near + z_far - 2 * z_near * z_far / depth) / (z_far - z_near)


def ndc_depth_to_depth(ndc_depth: Tensor, z_near: float, z_far: float, z_range=(-1, 1)):
    if z_range == (-1, 1):
        return 2 * z_near * z_far / (z_near + z_far - ndc_depth * (z_far - z_near))


def adjust_intrinsics_size(intrinsics: np.ndarray, width: int, height: int):
    """
        intrinsics: np.ndarray or Tensor [..., 3, 3]
    """
    width_raw, height_raw = intrinsics[..., 0, 2] * 2, intrinsics[..., 1, 2] * 2
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = height / 2
    intrinsics[..., 0, 0] *= width / width_raw
    intrinsics[..., 1, 1] *= height / height_raw
    return intrinsics


class RandomCamera:
    def __init__(self, device, cfg: DataConfig, image_height: int, image_width: int) -> None:
        # Basic
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        # Camera
        self.radius_range = cfg.radius_range  # (0, +inf)
        self.fovy_range = cfg.fovy_range  # (0, 180)
        self.azimuth_range = cfg.azimuth_range  # (0, 360)
        self.elevation_range = cfg.elevation_range  # (0, 180)
        self.z_far = cfg.z_far
        self.z_near = cfg.z_near
        # Augmentation
        self.vertical_jitter = cfg.vertical_jitter  # Tuple[float, float]
        # Progressive Training Setting
        self.training_ratio: float = None
        self.progressive_radius = cfg.progressive_radius
        self.progressive_radius_ranges = eval(str(cfg.progressive_radius_ranges))
        # Correction
        self.camera_offset = cfg.camera_offset  # Tuple[float, float, float]

    def get_radius(self, size):
        if self.progressive_radius:
            start_range, end_range = self.progressive_radius_ranges  # (2.5, 3.5) -> (1.0, 2.0)
            radius_begin = start_range[0] + self.training_ratio * (end_range[0] - start_range[0])
            radius_end = start_range[1] + self.training_ratio * (end_range[1] - start_range[1])
        else:
            radius_begin, radius_end = self.radius_range[0], self.radius_range[1]
        return torch.rand(size, device=self.device) * (radius_end - radius_begin) + radius_begin

    def get_angle(self, size, intervals):
        # single interval
        if len(intervals) == 2 and isinstance(intervals[0], Number) and isinstance(intervals[1], Number):
            a, b = intervals
        # multiple intervals
        else:
            total_weight = sum(b - a + 1e-12 for a, b in intervals)
            probabilities = [(b - a + 1e-12) / total_weight for a, b in intervals]
            chosen_interval_index = np.random.choice(len(intervals), p=probabilities)
            a, b = intervals[chosen_interval_index]
        return a + torch.rand(size, device=self.device) * (b - a)

    def to_extrinsic_with_jitter(self, radius, azimuth, elevation):
        # look at
        at_vector = torch.zeros(radius.shape[0], 3, device=radius.device)
        # offset
        if self.camera_offset is not None:
            at_vector += torch.tensor(self.camera_offset, device=self.device).unsqueeze(0)
        # vertical jitter
        if self.vertical_jitter is not None:
            vertical_camera_offset = np.random.uniform(*self.vertical_jitter)
            at_vector[..., 1] += vertical_camera_offset
        # to extrinsic
        return to_extrinsic(radius=radius, azimuth=azimuth, elevation=elevation, at_vector=at_vector)

    def __call__(self, size: int):
        ''' generate random poses from an orbit camera
        Args:
            size: int.
        Return:
            extrinsic: [size, 4, 4]
            intrinsics: [size, 3, 3]
        '''
        radius = self.get_radius(size)
        azimuth = self.get_angle(size, self.azimuth_range)
        elevation = self.get_angle(size, self.elevation_range)
        fov = self.get_angle(size, self.fovy_range)
        tanfov = get_tan_half_fov(fov)

        extrinsic, c2w = self.to_extrinsic_with_jitter(
            radius=radius,
            azimuth=azimuth,
            elevation=elevation,
        )

        intrinsics = to_intrinsics(
            tanfov=tanfov,
            image_height=self.image_height,
            image_width=self.image_width,
        )

        projection = to_projection(
            tanfov=tanfov,
            z_far=self.z_far,
            z_near=self.z_near,
        )

        mvp = torch.bmm(projection, extrinsic)

        screen = to_screen(
            mvp=mvp,
            image_height=self.image_height,
            image_width=self.image_width,
        )

        return {
            'extrinsic': extrinsic,
            'c2w': c2w,
            'intrinsics': intrinsics,
            'mvp': mvp,
            'projection': projection,
            'screen': screen,
            'azimuth': azimuth,
            'elevation': elevation,
            'radius': radius,
            'fov': fov,
            'tanfov': tanfov,
            'z_far': self.z_far,
            'z_near': self.z_near,
            'image_height': self.image_height,
            'image_width': self.image_width,
        }


class RandomCamera4Avatar(RandomCamera):
    KEYPOINT_NAMES = OPENPOSE_KEYPOINT_NAMES

    def __init__(self, device, cfg: DataConfig, image_height: int, image_width: int) -> None:
        super().__init__(
            device=device,
            cfg=cfg,
            image_height=image_height,
            image_width=image_width,
        )
        self.params = {
            'body': {'prob': cfg.body_prob, 'azimuth_range': self.azimuth_range, 'elevation_range': self.elevation_range, 'radius_range': self.radius_range},
            'head': {'prob': cfg.head_prob, 'azimuth_range': cfg.head_azimuth_range, 'elevation_range': cfg.head_elevation_range, 'radius_range': cfg.head_radius_range},
            'face': {'prob': cfg.face_prob, 'azimuth_range': cfg.face_azimuth_range, 'elevation_range': cfg.face_elevation_range, 'radius_range': cfg.face_radius_range},
            'hand_left': {'prob': cfg.hand_prob/2, 'azimuth_range': cfg.hand_left_azimuth_range, 'elevation_range': cfg.hand_elevation_range, 'radius_range': cfg.hand_radius_range},
            'hand_right': {'prob': cfg.hand_prob/2, 'azimuth_range': cfg.hand_right_azimuth_range, 'elevation_range': cfg.hand_elevation_range, 'radius_range': cfg.hand_radius_range},
            'foot_left': {'prob': cfg.foot_prob/2, 'azimuth_range': cfg.foot_left_azimuth_range, 'elevation_range': cfg.foot_elevation_range, 'radius_range': cfg.foot_radius_range},
            'foot_right': {'prob': cfg.foot_prob/2, 'azimuth_range': cfg.foot_right_azimuth_range, 'elevation_range': cfg.foot_elevation_range, 'radius_range': cfg.foot_radius_range},
            'arm_left': {'prob': cfg.arm_prob/2, 'azimuth_range': ((0, 360),), 'elevation_range': (75, 105), 'radius_range': (0.5, 1.0)},
            'arm_right': {'prob': cfg.arm_prob/2, 'azimuth_range': ((0, 360),), 'elevation_range': (75, 105), 'radius_range': (0.5, 1.0)},
        }
        self.params['body']['camera_offset'] = self.camera_offset
        self.vertical_jitter_copy = self.vertical_jitter
        self.progressive_radius_copy = self.progressive_radius
        self.use_human_vertical_jitter = cfg.use_human_vertical_jitter
        self.keys = sorted(list(self.params.keys()))

    def choice_body_part(self) -> str:
        total_weight = sum(self.params[k]['prob'] + 1e-12 for k in self.keys)
        probabilities = [(self.params[k]['prob'] + 1e-12) / total_weight for k in self.keys]
        chosen_part_index = np.random.choice(len(self.keys), p=probabilities)
        return self.keys[chosen_part_index]

    def setup_camera_offset(self, keypoints: np.ndarray, verbose: bool = False):
        """
        keypoints: np.ndarray, [1, 18, 3]
        keypoints_names = [
            'nose', 'neck',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
            'right_eye', 'left_eye', 'right_ear', 'left_ear',
        ]
        """
        assert keypoints.shape[0] == 1 and keypoints.ndim == 3
        # Body
        if self.use_human_vertical_jitter:
            vertical_jitter = [
                (
                    # keypoints[0, self.KEYPOINT_NAMES['left_knee']][1] +
                    # keypoints[0, self.KEYPOINT_NAMES['right_knee']][1]
                    keypoints[0, self.KEYPOINT_NAMES['left_ankle']][1] +
                    keypoints[0, self.KEYPOINT_NAMES['right_ankle']][1]
                ) / 2,
                (
                    keypoints[0, self.KEYPOINT_NAMES['left_shoulder']][1] +
                    keypoints[0, self.KEYPOINT_NAMES['right_shoulder']][1]
                ) / 2,
            ]
            self.vertical_jitter_copy = vertical_jitter
            if verbose:
                logger.info(f'Use adaptive vertical jitter {vertical_jitter} that infers from human template!')
        # Head
        self.params['head']['camera_offset'] = (
            keypoints[0, self.KEYPOINT_NAMES['left_ear']] +
            keypoints[0, self.KEYPOINT_NAMES['right_ear']]
        ) / 2.0
        # Face
        self.params['face']['camera_offset'] = (
            keypoints[0, self.KEYPOINT_NAMES['left_ear']] +
            keypoints[0, self.KEYPOINT_NAMES['right_ear']]
        ) / 2.0
        # Arm
        self.params['arm_left']['camera_offset'] = (
            keypoints[0, self.KEYPOINT_NAMES['left_elbow']] * 1 / 3 +
            keypoints[0, self.KEYPOINT_NAMES['left_wrist']] * 2 / 3
        )
        self.params['arm_right']['camera_offset'] = (
            keypoints[0, self.KEYPOINT_NAMES['right_elbow']] * 1 / 3 +
            keypoints[0, self.KEYPOINT_NAMES['right_wrist']] * 2 / 3
        )
        # Foot
        self.params['foot_left']['camera_offset'] = keypoints[0, self.KEYPOINT_NAMES['left_ankle']] + np.array([0.0, -0.05, 0.0])
        self.params['foot_right']['camera_offset'] = keypoints[0, self.KEYPOINT_NAMES['right_ankle']] + np.array([0.0, -0.05, 0.0])
        # Hand
        if keypoints.shape[1] == 18:
            # smpl
            self.params['hand_left']['camera_offset'] = keypoints[0, self.KEYPOINT_NAMES['left_wrist']] + np.array([0.0, -0.1, 0.0])
            self.params['hand_right']['camera_offset'] = keypoints[0, self.KEYPOINT_NAMES['right_wrist']] + np.array([0.0, -0.1, 0.0])
        else:
            # smplx or smplh
            self.params['hand_left']['camera_offset'] = (
                keypoints[0, self.KEYPOINT_NAMES['left_wrist_new']] +
                keypoints[0, self.KEYPOINT_NAMES['left_middle1']] +
                keypoints[0, self.KEYPOINT_NAMES['left_middle2']] +
                keypoints[0, self.KEYPOINT_NAMES['left_middle3']] +
                keypoints[0, self.KEYPOINT_NAMES['left_middle']]
            ) / 5.0
            self.params['hand_right']['camera_offset'] = (
                keypoints[0, self.KEYPOINT_NAMES['right_wrist_new']] +
                keypoints[0, self.KEYPOINT_NAMES['right_middle1']] +
                keypoints[0, self.KEYPOINT_NAMES['right_middle2']] +
                keypoints[0, self.KEYPOINT_NAMES['right_middle3']] +
                keypoints[0, self.KEYPOINT_NAMES['right_middle']]
            ) / 5.0

    def __call__(self, size: int, body_part: Optional[str] = None):
        if body_part is None:
            chosen_body_part = self.choice_body_part()
        else:
            chosen_body_part = body_part

        self.azimuth_range = self.params[chosen_body_part]['azimuth_range']
        self.elevation_range = self.params[chosen_body_part]['elevation_range']
        self.radius_range = self.params[chosen_body_part]['radius_range']
        self.camera_offset = self.params[chosen_body_part]['camera_offset']

        if chosen_body_part == 'body':
            self.progressive_radius = self.progressive_radius_copy
            self.vertical_jitter = self.vertical_jitter_copy
        else:
            self.progressive_radius = False
            self.vertical_jitter = None
        res = super().__call__(size)
        res['body_part'] = chosen_body_part
        return res


class CyclicalCamera:
    def __init__(self, device, cfg: DataConfig, image_height: int, image_width: int) -> None:
        # Basic
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        # Camera
        self.radius: float = cfg.eval_radius if cfg.eval_radius else max(cfg.radius_range) * cfg.eval_radius_rate
        self.azimuth: float = cfg.eval_azimuth
        self.elevation: float = cfg.eval_elevation
        self.fov: float = (cfg.fovy_range[0] + cfg.fovy_range[1]) / 2
        # Track
        self.camera_offset = cfg.eval_camera_offset
        self.trajectory = cfg.eval_camera_track

    def __call__(self, p: float, **kwargs):
        camera_params = {
            'device': self.device,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'radius': self.radius,
            'azimuth': self.azimuth,
            'elevation': self.elevation,
            'fov': self.fov,
            'camera_offset': self.camera_offset,
            'trajectory': self.trajectory,
        }
        camera_params.update(kwargs)
        return cyclical_camera(p=p, **camera_params)


class CyclicalCamera4Avatar(CyclicalCamera):
    KEYPOINT_NAMES = OPENPOSE_KEYPOINT_NAMES

    def __init__(self, device, cfg: DataConfig, image_height: int, image_width: int) -> None:
        super().__init__(device, cfg, image_height, image_width)
        self.default_body_part = cfg.eval_body_part
        self.chosen_body_part = None
        if self.camera_offset is None:
            self.default_camera_offset = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        else:
            self.default_camera_offset = np.array(self.camera_offset, dtype=np.float32)

    def setup_camera_offset(self, keypoints: np.ndarray, verbose: bool = False, body_part: Optional[str] = None):
        """
        keypoints: np.ndarray, [1, 18, 3]
        keypoints_names = [
            'nose', 'neck',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
            'right_eye', 'left_eye', 'right_ear', 'left_ear',
        ]
        """
        if body_part is None:
            chosen_body_part = self.default_body_part
        else:
            chosen_body_part = body_part

        # assert keypoints.shape[0] == 1 and keypoints.ndim == 3
        assert keypoints.ndim == 3

        if chosen_body_part == 'head' or chosen_body_part == 'face':
            self.camera_offset = self.default_camera_offset + (
                keypoints[0, self.KEYPOINT_NAMES['left_ear']] +
                keypoints[0, self.KEYPOINT_NAMES['right_ear']]
            ) / 2.0
        
        elif chosen_body_part == 'left_hand':

            if keypoints.shape[1] == 18:
                # smpl
                self.camera_offset = self.default_camera_offset + \
                    keypoints[0, self.KEYPOINT_NAMES['left_wrist']] + np.array([0.0, -0.1, 0.0])
            else:
                # smplx or smplh
                self.camera_offset = self.default_camera_offset + (
                    keypoints[0, self.KEYPOINT_NAMES['left_wrist_new']] +
                    keypoints[0, self.KEYPOINT_NAMES['left_middle1']] +
                    keypoints[0, self.KEYPOINT_NAMES['left_middle2']] +
                    keypoints[0, self.KEYPOINT_NAMES['left_middle3']] +
                    keypoints[0, self.KEYPOINT_NAMES['left_middle']]
                ) / 5.0

        elif chosen_body_part == 'right_hand':

            if keypoints.shape[1] == 18:
                # smpl
                self.camera_offset = self.default_camera_offset + \
                    keypoints[0, self.KEYPOINT_NAMES['right_wrist']] + np.array([0.0, -0.1, 0.0])
            else:
                # smplx or smplh
                self.camera_offset = self.default_camera_offset + (
                    keypoints[0, self.KEYPOINT_NAMES['right_wrist_new']] +
                    keypoints[0, self.KEYPOINT_NAMES['right_middle1']] +
                    keypoints[0, self.KEYPOINT_NAMES['right_middle2']] +
                    keypoints[0, self.KEYPOINT_NAMES['right_middle3']] +
                    keypoints[0, self.KEYPOINT_NAMES['right_middle']]
                ) / 5.0
        
        elif chosen_body_part in self.KEYPOINT_NAMES.keys():
            self.camera_offset = self.default_camera_offset + keypoints[0, self.KEYPOINT_NAMES[chosen_body_part]]

        else:
            assert chosen_body_part in (None, 'body')
        
        self.chosen_body_part = chosen_body_part
        
        if verbose:
            logger.info(f'Setup CyclicalCamera4Avatar with body part: {chosen_body_part} and camera offset: {self.camera_offset}')

    def __call__(self, p: float):
        res = super().__call__(p)
        res['body_part'] = self.chosen_body_part
        return res


def sample_camera_trajectory(
    p: float,
    azimuth: float = 0.0,
    elevation: float = 90.0,
    trajectory: str = 'circle',
):
    if trajectory == 'fixed':
        return azimuth, elevation
    
    elif trajectory == 'circle':
        azimuth = p * 360
        elevation = elevation

    elif trajectory == 'wave-elev':
        azimuth = p * 360
        elevation = np.sin(p * 2 * np.pi) * 30
    
    elif trajectory == 'wave':
        azimuth += np.sin(p * 4 * np.pi) * 20
        elevation += np.cos(p * 4 * np.pi) * 10
        azimuth %= 360.0
        elevation %= 360.0

    else:
        raise ValueError(f'Unknown trajectory: {trajectory}')
    
    return azimuth, elevation


def cyclical_camera(
    p: float,
    device='cpu',
    image_height=512,
    image_width=512,
    radius=2.2,
    elevation=90.0,
    azimuth=0.0,
    fov=(40+70)/2,
    camera_offset=None,
    trajectory='circle',
    z_near=0.01,
    z_far=1000.,
):
    # sample trajectory
    azimuth, elevation = sample_camera_trajectory(
        p=p,
        azimuth=azimuth,
        elevation=elevation,
        trajectory=trajectory,
    )

    # prepare
    radius = torch.FloatTensor([radius,]).to(device)
    azimuth = torch.FloatTensor([azimuth,]).to(device)
    elevation = torch.FloatTensor([elevation,]).to(device)
    fov = torch.FloatTensor([fov,]).to(device)
    tanfov = get_tan_half_fov(fov)

    aspect_ratio = aspect_wh = image_width / image_height
    tanfov_y = tanfov
    tanfov_x = tanfov_y * aspect_wh

    at_vector = torch.zeros(radius.shape[0], 3, device=device)
    if camera_offset is not None:
        at_vector += torch.tensor(camera_offset, device=device).unsqueeze(0)

    # caculate
    extrinsic, c2w = to_extrinsic(
        radius=radius,
        azimuth=azimuth,
        elevation=elevation,
        at_vector=at_vector,
    )
    intrinsics = to_intrinsics(
        tanfov=tanfov,
        image_height=image_height,
        image_width=image_width,
    )
    projection = to_projection(
        tanfov=tanfov,
        aspect_wh=aspect_wh,
        z_far=z_far,
        z_near=z_near,
    )
    mvp = torch.bmm(projection, extrinsic)


    # return
    return {
        'extrinsic': extrinsic,
        'c2w': c2w,
        'intrinsics': intrinsics,
        'mvp': mvp,
        'projection': projection,
        'azimuth': azimuth,
        'elevation': elevation,
        'aspect_ratio': aspect_ratio,
        'radius': radius,
        'fov': fov,
        'tanfov': tanfov,
        'tanfov_x': tanfov_x,
        'tanfov_y': tanfov_y,
        'z_far': z_far,
        'z_near': z_near,
        'image_height': image_height,
        'image_width': image_width,
    }


def visualize_camera(c2w: Tensor, dirs=None, size=0.2, sphere_radius=0.05, draw_axis:bool=True):
    """
    Args:
        c2w: torch.Tensor, [B, 4, 4]
        XYZ <-> RGB
    """
    DIR_COLORS = np.array([
    [0, 0, 0, 255], # default
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
    ], dtype=np.uint8)

    import trimesh
    axes = trimesh.creation.axis(axis_length=1)
    sphere = trimesh.creation.icosphere(radius=sphere_radius)
    objects = [axes, sphere]

    c2w = c2w.detach().cpu().numpy()

    if dirs is None:
        dirs = np.zeros((c2w.shape[0],), dtype=np.int8)

    for pose, dir in zip(c2w, dirs):
        # a camera is visualized with 8 line segments.
        # pose: np.ndarray, [4, 4]
        # pos: np.ndarray, [3, ]
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a],])
        segs = trimesh.load_path(segs)
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)  # different color for different dirs
        objects.append(segs)

        if draw_axis:
            x = (pose @ np.array([[0.5], [0.0], [0.0], [1.0]]))[:3, 0]
            y = (pose @ np.array([[0.0], [0.5], [0.0], [1.0]]))[:3, 0]
            z = (pose @ np.array([[0.0], [0.0], [5.0], [1.0]]))[:3, 0]
            x_axis = trimesh.load_path(np.array([[pos, x],]), colors=np.array([[255, 0, 0, 255],]))
            y_axis = trimesh.load_path(np.array([[pos, y],]), colors=np.array([[0, 255, 0, 255],]))
            z_axis = trimesh.load_path(np.array([[pos, z],]), colors=np.array([[0, 0, 255, 255],]))
            objects.append(x_axis)
            objects.append(y_axis)
            objects.append(z_axis)

    # trimesh.Scene(objects).show()
    return objects


def visualize_camera_with_smpl(c2w: Tensor, smpl_vertices: np.ndarray, smpl_faces: np.ndarray, **kwargs):
    """
    Args:
        c2w: torch.Tensor, [B, 4, 4]
        smpl_vertices: np.ndarray, [V, 3]
        smpl_faces: np.ndarray, [F, 3]
        XYZ <-> RGB
    """
    import trimesh
    if isinstance(smpl_vertices, Tensor):
        smpl_vertices = smpl_vertices.detach().cpu().numpy()
        smpl_vertices = smpl_vertices.squeeze(0)
    if isinstance(smpl_faces, Tensor):
        smpl_faces = smpl_faces.detach().cpu().numpy()
    vertex_colors = np.ones([smpl_vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    smpl_meshs = trimesh.Trimesh(smpl_vertices, smpl_faces, vertex_colors=vertex_colors)
    objects = visualize_camera(c2w, **kwargs)
    objects.append(smpl_meshs)

    # trimesh.Scene(objects).show()
    return objects
