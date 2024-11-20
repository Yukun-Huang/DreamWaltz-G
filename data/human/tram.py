import os.path as osp
import torch
import numpy as np
from pytorch3d.transforms import matrix_to_axis_angle
from configs.paths import TRAM_ROOT


class Tram(object):
    """Tram class for loading and processing SMPL/Camera parameters from TRAM 3D body estimation."""
    def __init__(self, root: str = TRAM_ROOT) -> None:
        self.root = root

    def convert_camera_params(self, camera_data, z_near=0.01, z_far=1000.0):
        """
        Convert TRAM camera parameters to format compatible with dreamwaltz-g.
        """
        # Number of frames
        num_frames = camera_data['pred_cam_R'].shape[0]

        # Create full extrinsic matrices: [N, 4, 4]
        extrinsic = np.eye(4).reshape(1, 4, 4).repeat(num_frames, axis=0)
        # extrinsic[:, :3, :3] = camera_data['pred_cam_R']  # [N, 3, 3]
        # extrinsic[:, :3, 3] = camera_data['pred_cam_T']  # [N, 3]

        # Flip y axis
        extrinsic[:, 1, :] *= -1

        f = camera_data['img_focal']
        cx = int(camera_data['img_center'][0])
        cy = int(camera_data['img_center'][1])

        image_width = cx * 2 
        image_height = cy * 2
        aspect_ratio = image_width / image_height

        intrinsics = np.zeros((num_frames, 3, 3))
        intrinsics[:, 0, 0] = f
        intrinsics[:, 1, 1] = f
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        intrinsics[:, 2, 2] = 1.0

        tanfov_y = np.full(num_frames, cy / f)
        tanfov_x = np.full(num_frames, cx / f)
        fov_x = np.full(num_frames, np.degrees(2 * np.arctan(tanfov_x[0])))
        fov_y = np.full(num_frames, np.degrees(2 * np.arctan(tanfov_y[0])))

        # Final returned structure
        return {
            'extrinsic': extrinsic,
            'intrinsics': intrinsics,
            'z_far': z_far,
            'z_near': z_near,
            'fov': fov_y,
            'fov_x': fov_x,
            'fov_y': fov_y,
            'tanfov': tanfov_y,
            'tanfov_y': tanfov_y,
            'tanfov_x': tanfov_x,
            'aspect_ratio': aspect_ratio,
            'image_height': image_height,
            'image_width': image_width,
            'pred_cam_R': camera_data['pred_cam_R'],
            'pred_cam_T': camera_data['pred_cam_T'],
        }

    def get_smpl_params(self, filename: str, model_type: str = 'smplx'):
        # Check model type
        assert model_type == 'smplx', f"Model type {model_type} not supported for TRAM dataset."

        # Load SMPL and camera parameters
        # TODO: Add support for multiple characters
        smpl_path = osp.join(self.root, filename, f'animation/hps_track_0.npy')
        camera_path = osp.join(self.root, filename, f'camera/camera.npy')

        camera_data = np.load(camera_path, allow_pickle=True).item()
        smpl_data = np.load(smpl_path, allow_pickle=True).item()
   
        # Video in the wild dimensions
        camera_params = self.convert_camera_params(camera_data)

        # Convert torch tensors to numpy if needed
        pred_rotmat = smpl_data['pred_rotmat'].numpy() if torch.is_tensor(smpl_data['pred_rotmat']) else smpl_data['pred_rotmat']
        pred_shape = smpl_data['pred_shape'].numpy() if torch.is_tensor(smpl_data['pred_shape']) else smpl_data['pred_shape']
        pred_trans = smpl_data['pred_trans'].numpy() if torch.is_tensor(smpl_data['pred_trans']) else smpl_data['pred_trans']

        num_frames = pred_rotmat.shape[0]

        # Convert rotation matrices to axis-angle
        body_pose_rotmat = pred_rotmat[:, 1:24]  # exclude global orient

        # Convert to axis-angle using pytorch3d
        body_pose_rotmat_torch = torch.from_numpy(body_pose_rotmat)
        body_pose = matrix_to_axis_angle(body_pose_rotmat_torch).numpy()
        global_orient = matrix_to_axis_angle(torch.from_numpy(pred_rotmat[:, 0:1])).numpy()

        # Create SMPL-X compatible dictionary
        smplx_params_dict = {
            # Global orientation
            'global_orient': global_orient,  # [N, 1, 3]

            # Body pose (convert 23 SMPL joints to 21 SMPL-X body joints)
            'body_pose': body_pose[:, :21].reshape(num_frames, -1),  # [N, 63]

            # Face parameters (set to neutral)
            'jaw_pose': np.zeros((num_frames, 1, 3)),      # [N, 1, 3]
            'leye_pose': np.zeros((num_frames, 1, 3)),     # [N, 1, 3]
            'reye_pose': np.zeros((num_frames, 1, 3)),     # [N, 1, 3]

            # Hand poses (set to relaxed pose)
            'left_hand_pose': np.zeros((num_frames, 15, 3)),   # [N, 15, 3]
            'right_hand_pose': np.zeros((num_frames, 15, 3)),  # [N, 15, 3]

            # Shape and translation
            'betas': pred_shape,              # [N, 10]
            'transl': pred_trans.squeeze(1)   # [N, 3]
        }

        # Add batch dimension
        smplx_params = {
            key: value[np.newaxis, ...] for key, value in smplx_params_dict.items()
        }
        
        return smplx_params, camera_params
