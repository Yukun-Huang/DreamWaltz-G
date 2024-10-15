import numpy as np
from PIL import Image
from typing import Any, Iterable, Optional
from smplx.utils import SMPLXOutput
import torch
import nvdiffrast.torch as dr
from .open_pose import PoseResult, BodyResult, Keypoint
from .open_pose import adaptive_draw_poses as draw_poses
# from .open_pose import draw_poses

from configs import PromptConfig
from utils.trimesh import render_trimesh, to_trimesh
from utils.point3d import *
from utils.mesh import export_normal_nvdiffrast
from utils.se3 import SE3_Mat2RT
from data.camera.utils import adjust_intrinsics_size
import utils.open3d as o3d_utils


def to_controlnet_pose(predicts, intrinsics, include_hand=True, include_face=True, distances=None):
    """
      predicts: np.array, [N, J, 2], J = 128 (body: 18, left hand: 21, right hand: 21, face landmarks: 51, face_contour: 17)
      distances: np.array, [N, J], J = 128 (body: 18, left hand: 21, right hand: 21, face landmarks: 51, face_contour: 17)
      https://github.com/vchoutas/smplify-x/issues/152
    """
    W, H = intrinsics[0, 2] * 2, intrinsics[1, 2] * 2

    def _construct_keypoint(pt_xy, d=-1.0):
        x, y = pt_xy[0], pt_xy[1]
        if np.isnan(x) or np.isnan(y):
        # if np.isnan(x) or np.isinf(x) or x < 0.0 \
        #   or np.isnan(y) or np.isinf(y) or y < 0.0:
            return None
        else:
            return Keypoint(x = x / float(W), y = y / float(H), dist=d)

    results = []
    if distances is None:
        distances = np.ones_like(predicts[:, :, 0]) * -1.0

    for pred, dist in zip(predicts, distances):
        # body
        keypoints = []
        for i in range(18):
            keypoints.append(_construct_keypoint(pred[i], dist[i]))
        start_idx = 18
        # others
        left_hand, right_hand, face = (None,) * 3
        if include_hand and len(pred) > 18:
            # left hand
            left_hand = []
            for i in range(21):
                i += start_idx
                left_hand.append(_construct_keypoint(pred[i], dist[i]))
            start_idx += 21
            # right hand
            right_hand = []
            for i in range(21):
                i += start_idx
                right_hand.append(_construct_keypoint(pred[i], dist[i]))
            start_idx += 21
        # face
        if include_face and len(pred) > 18 + 21 + 21:
            face = []
            # landmarks
            for i in range(51):
                i += start_idx
                face.append(_construct_keypoint(pred[i], dist[i]))
            start_idx += 51
            # contour
            for i in range(17):
                i += start_idx
                face.append(_construct_keypoint(pred[i], dist[i]))
            start_idx += 17
        # results
        body_results = BodyResult(keypoints=keypoints, total_score=1.0, total_parts=-1,)
        results.append(PoseResult(body_results, left_hand, right_hand, face))

    return results


class OcclusionCulling:
    def __init__(self, smpl_type:str, ignore_body_self_occlusion:bool=False) -> None:
        if smpl_type == 'smpl':
            self.face_indices = [0, 14, 15, 16, 17]
            self.hand_indices = []
            self.body_indices = [i for i in range(18) if i not in self.face_indices]
        elif smpl_type == 'smplx':
            self.face_indices = [0, 14, 15, 16, 17] + [i for i in range(18 + 21 * 2, 128)]
            self.hand_indices = [i for i in range(18, 18 + 21 * 2)]
            self.body_indices = [i for i in range(128) if (i not in self.face_indices) and (i not in self.hand_indices)]
        else:
            raise NotImplementedError
        self.ignore_body_self_occlusion = ignore_body_self_occlusion

    def __call__(
        self,
        center: np.ndarray,
        keypoints: np.ndarray,
        ray_casting_scene,
        thres_body=0.2,
        thres_face=0.02,
        # thres_hand=0.05,
        thres_hand=0.2,
    ):
        """
        Input:
            center: np.array, [3, 1], camera position in world coordinates
            keypoints: np.array, [N, K, 3], keypoints in world coordinates
        Return:
            occluded: np.array, bool, [N, K]
        """
        center = np.broadcast_to(center.T, (*(keypoints.shape[:-1]), 3))  # [N, K, 3]
        t_far = np.linalg.norm(keypoints - center, ord=2, axis=2)            # [N, K]

        directions = keypoints - center
        directions /= np.linalg.norm(directions, ord=2, axis=2, keepdims=True)  # [N, K, 3]
        rays = np.concatenate((center, directions), axis=2)   # [N, K, 6]

        outputs = ray_casting_scene.cast_rays(np.asarray(rays, dtype=np.float32))
        t_hit = outputs['t_hit'].numpy()                # [N, K]
        geometry_ids = outputs['geometry_ids'].numpy()  # [2, K], int

        # Face Occlusion
        occluded_face = (t_far[:, self.face_indices] - t_hit[:, self.face_indices]) > thres_face  # [N, 5], bool

        # Hand Occlusion
        occluded_hand = (t_far[:, self.hand_indices] - t_hit[:, self.hand_indices]) > thres_hand  # [N, 42], bool

        # Body Occlusion
        occluded_body = (t_far[:, self.body_indices] - t_hit[:, self.body_indices]) > thres_body  # [N, 13], bool
        if self.ignore_body_self_occlusion:
            self_geometry_ids = np.array([[i for i in range(len(keypoints))]]).T  # [N, 1]
            self_occluded_body = geometry_ids[:, self.body_indices] == self_geometry_ids  # [N, 13], bool
            occluded_body = occluded_body & (~ self_occluded_body)  # [N, 13], bool

        # Return
        occluded = np.zeros_like(t_hit, dtype=np.bool_)
        occluded[:, self.face_indices] = occluded_face
        occluded[:, self.hand_indices] = occluded_hand
        occluded[:, self.body_indices] = occluded_body

        return occluded, t_far


class SMPL2Condition:
    def __init__(self, cfg: PromptConfig) -> None:
        self.glctx = None
        self.draw_body = cfg.draw_body_keypoints
        self.draw_hand = cfg.draw_hand_keypoints
        self.draw_face = cfg.draw_face_landmarks
        # self.hand_dist_thres = cfg.adaptive_hand_dist_thres
        self.openpose_left_right_flip = cfg.openpose_left_right_flip
        if cfg.use_occlusion_culling:
            self.occlusion_culling = OcclusionCulling(cfg.smpl_type, cfg.ignore_body_self_occlusion)
        else:
            self.occlusion_culling = None

    def export_2d_keypoints(self, keypoints, **camera_params):
        """
        Input:
            keypoints: np.array, [N, K, 3]
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        """
        extrinsic = camera_params['extrinsic'].cpu().numpy()[0]
        intrinsics = camera_params['intrinsics'].cpu().numpy()[0]
        assert extrinsic.ndim == 2 and extrinsic.size == 16
        assert intrinsics.ndim == 2 and intrinsics.size == 9
        intrinsics = adjust_intrinsics_size(
            intrinsics=intrinsics,
            width=camera_params['image_width'],
            height=camera_params['image_height']
        )
        width: int = camera_params['image_width']
        height: int = camera_params['image_height']
        # Init
        N, K, _ = keypoints.shape
        R, T = SE3_Mat2RT(extrinsic)
        # World-to-Camera Transform
        keypoints_cam = transform_keypoints_to_novelview(keypoints.reshape(-1, 3), None, None, R, T)
        invisible_indices = keypoints_cam[:, 2] < 0  # be carefull for z-axis definition
        keypoints_cam[invisible_indices] = None
        # Camera-to-Image Transform
        keypoints_img = project_camera3d_to_2d(keypoints_cam, intrinsics)  # [N*K, 2]
        keypoints_img = keypoints_img.reshape(N, K, 2)  # [N, K, 2]
        keypoints_img[:, :, 0] /= float(width)
        keypoints_img[:, :, 1] /= float(height)
        return keypoints_img

    def export_pose(self, keypoints, ray_casting_scene, **camera_params):
        """
        Input:
            keypoints: np.array, [N, K, 3]
            intrinsics: np.array, [3, 3]
            extrinsic: np.array, [4, 4], world -> camera
        """
        # Init
        extrinsic: np.ndarray = camera_params['extrinsic']
        intrinsics: np.ndarray = camera_params['intrinsics']
        width: int = camera_params['width']
        height: int = camera_params['height']
        # Init
        N, K, _ = keypoints.shape
        R, T = SE3_Mat2RT(extrinsic)
        # World-to-Camera Transform
        keypoints_cam = transform_keypoints_to_novelview(keypoints.reshape(-1, 3), None, None, R, T)
        invisible_indices = keypoints_cam[:, 2] < 0  # be carefull for z-axis definition
        keypoints_cam[invisible_indices] = None
        # Camera-to-Image Transform
        keypoints_img = project_camera3d_to_2d(keypoints_cam, intrinsics)  # [N*K, 2]
        keypoints_img = keypoints_img.reshape(N, K, 2)  # [N, K, 2]
        # Occlusion Culling
        if self.occlusion_culling is not None:
            center = np.dot(np.linalg.inv(R), - T)
            occluded, distances = self.occlusion_culling(
                center=center,
                keypoints=keypoints,
                ray_casting_scene=ray_casting_scene,
            )  # [N, K]
            keypoints_img[occluded, :] = np.nan  # [N, K, 2]
        else:
            distances = None
        # Draw
        controlnet_pose = to_controlnet_pose(keypoints_img, intrinsics=intrinsics, distances=distances)
        image = draw_poses(
            poses=controlnet_pose,
            H=height,
            W=width,
            draw_body=self.draw_body,
            draw_hand=self.draw_hand,
            draw_face=self.draw_face,
            flip_LR=self.openpose_left_right_flip,
        )
        return Image.fromarray(image)

    def export_depth(self, ray_casting_scene, inverse=True, normalize=True, raw=False, **camera_params):
        depth = o3d_utils.cast_rays(ray_casting_scene, return_type='depth', **camera_params)
        if raw:
            return depth  # np.ndarray, [H, W]
        # Inverse and Normalize
        if inverse:
            depth = 1.0 / depth
        if normalize:
            depth -= np.min(depth)
            depth /= np.max(depth)
        image = np.asarray(depth * 255.0, np.uint8)
        image = np.stack([image, image, image], axis=2)
        return Image.fromarray(image)

    def export_normal(self, vertices, triangles, mvp, h, w):
        """
        Args:
            vertices: vertices, torch.Tensor, [V, 3]
            f: faces, torch.Tensor, [F, 3]
            mvp: torch.Tensor, [B, 4, 4]
        """
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext()
        normal = export_normal_nvdiffrast(vertices, triangles, mvp=mvp, h=h, w=w, glctx=self.glctx, same_scene=True)
        normal = np.asarray(normal.detach().cpu().numpy() * 255.0, np.uint8)
        return Image.fromarray(normal)
    
    def export_normal_raw(self, ray_casting_scene, raw=False, **camera_params):
        normal = o3d_utils.cast_rays(ray_casting_scene, return_type='normal', **camera_params)
        if raw:
            return normal  # np.ndarray, [H, W, 3]
        image = np.asarray(normal * 255.0, np.uint8)
        return Image.fromarray(image)

    def __call__(
        self,
        smpl_outputs: SMPLXOutput,
        triangles: np.ndarray,
        camera_params: dict,
        condition_type: str,
        condition_height: int,
        condition_width: int,
    ):
        # Pytorch Style
        if condition_type == 'normal':
            vertices = smpl_outputs.vertices.detach()  # [B, V, 3]
            triangles = torch.tensor(triangles.astype(np.int32), device=vertices.device)
            return self.export_normal(
                vertices=vertices,
                triangles=triangles,
                mvp=camera_params['mvp'],
                h=condition_height,
                w=condition_width,
            )
        # Numpy Style
        else:
            vertices = smpl_outputs.vertices.detach().cpu().numpy()  # [B, V, 3]
            extrinsic = camera_params['extrinsic'].cpu().numpy()[0]
            intrinsics = camera_params['intrinsics'].cpu().numpy()[0]
            assert extrinsic.ndim == 2 and extrinsic.size == 16
            assert intrinsics.ndim == 2 and intrinsics.size == 9
            intrinsics = adjust_intrinsics_size(intrinsics=intrinsics, width=condition_width, height=condition_height)
            camera_params = {
                'intrinsics': intrinsics,
                'extrinsic': extrinsic,
                'width': condition_width,
                'height': condition_height,
            }
            # Render Condition
            if condition_type in ('pose', 'openpose'):
                keypoints = smpl_outputs.joints.detach().cpu().numpy()  # [B, J, 3]
                ray_casting_scene = o3d_utils.build_ray_casting_scene(vertices, triangles)
                return self.export_pose(keypoints, ray_casting_scene, **camera_params)
            elif condition_type == 'depth':
                ray_casting_scene = o3d_utils.build_ray_casting_scene(vertices, triangles)
                return self.export_depth(ray_casting_scene, **camera_params)
            elif condition_type == 'depth_raw':
                ray_casting_scene = o3d_utils.build_ray_casting_scene(vertices, triangles)
                return self.export_depth(ray_casting_scene, raw=True, **camera_params)
            elif condition_type == 'mesh':
                mesh = to_trimesh(vertices=vertices.squeeze(0), faces=triangles)
                return render_trimesh(mesh, **camera_params)
            else:
                assert 0, condition_type
