import os.path as osp
import numpy as np
from typing import Any


class Demo(object):
    def __init__(self, root: str = './assets/motions') -> None:
        self.root = root

    def get_smpl_params(self, filename:str, model_type:str='smplx'):

        assert model_type == 'smplx'

        filepath = osp.join(self.root, f'{filename}.npy')
        smplx_params = np.load(filepath)
        
        return {
            'jaw_pose': smplx_params[np.newaxis, :, 0:3],               # 3
            'global_orient': smplx_params[np.newaxis, :, 9:12],         # 3
            'body_pose': smplx_params[np.newaxis, :, 12:75],            # 63
            'left_hand_pose': smplx_params[np.newaxis, :, 75:120],      # 45
            'right_hand_pose': smplx_params[np.newaxis, :, 120:165],    # 45
            'expression': smplx_params[np.newaxis, :, 165:265],         # 100
        }
