import numpy as np
import os
import os.path as osp
import pickle as pkl
from glob import glob
import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from configs.paths import HYBRIK_ROOT


class Hybrik(object):
    def __init__(self, root: str = HYBRIK_ROOT) -> None:
        self.root = root
        filepaths_sp = glob(osp.join(root, 'SP', '*.pk'))
        filepaths_mp = glob(osp.join(root, 'MP', '*.pk'))
        filepaths = filepaths_sp + filepaths_mp
        dat = {}
        for fp in filepaths:
            fn = osp.splitext(osp.split(fp)[-1])[0]
            dat[fn] = fp
        self.dat = dat

    def get_video_info(self, transl):
        num_frame = len(transl)
        num_person_per_frame = [len(item) for item in transl]
        cnt = np.bincount(num_person_per_frame)
        num_person = cnt.argmax()
        return num_frame, num_person

    def get_smpl_params(self, filename, model_type='smpl', to_axis_angle=True):
        bdata = pkl.load(open(self.dat[filename], 'rb'))

        num_frame, num_person = self.get_video_info(bdata['transl'])

        betas_raw = bdata['pred_betas']
        poses_raw = bdata['pred_thetas']
        transl_raw = bdata['transl']

        assert len(betas_raw) == len(poses_raw) == len(transl_raw)

        betas, poses, transl = [], [], []
        for _betas, _poses, _transl in zip(betas_raw, poses_raw, transl_raw):
            if len(_betas) != num_person or len(_poses) != num_person or len(_transl) != num_person:
                continue
            betas.append(_betas)
            poses.append(_poses)
            transl.append(_transl)
        betas = np.array(betas).swapaxes(0, 1)
        poses = np.array(poses).swapaxes(0, 1)
        transl = np.array(transl).swapaxes(0, 1)

        num_frame = transl.shape[1]

        betas = np.mean(betas, axis=1, keepdims=False)
        poses = poses.reshape(num_person, num_frame, 24, 3, 3)  # [N, F, 24, 3, 3]

        if model_type in ('smplx', 'smplh'):
            num_joints = 21
        else:
            num_joints = 23

        if to_axis_angle:
            poses = matrix_to_axis_angle(torch.from_numpy(poses)).numpy()  # [N, F, 24, 3, 3] -> [N, F, 24, 3]
            global_orient = poses[:, :, 0, :]
            body_pose = poses[:, :, 1:(num_joints+1), :].reshape(num_person, num_frame, -1)
            # to zero
            global_orient = np.zeros_like(global_orient)
        else:
            global_orient = poses[:, :, 0, :, :]
            body_pose = poses[:, :, 1:(num_joints+1), :, :]

        smpl_params = {
            'global_orient': global_orient,  # (2, F, 3)
            'body_pose': body_pose,          # (2, F, 63/69)
            'betas': betas,                  # (2, 10)
            'transl': transl,                # (2, F, 3)
        }

        return smpl_params
