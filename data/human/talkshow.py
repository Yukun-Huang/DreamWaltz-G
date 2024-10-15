import torch
import numpy as np
import tarfile
from typing import Any
import os
import os.path as osp
import pickle as pkl
from glob import glob
from typing import Iterable, Optional
from collections import defaultdict
from configs.paths import TALKSHOW_ROOT


class TalkShow(object):
    def __init__(self, root: str = TALKSHOW_ROOT) -> None:
        self.root = root
        self.archives = None  # Lazy initialization
        self.pkl_files = None  # Lazy initialization
    
    def initialize(self):
        archives = {}
        pkl_files = {}
        for speaker in ("oliver", "seth", "chemistry", "conan"):
            tar_file = osp.join(self.root, f"{speaker}_pkl_tar.tar.gz")
            if not osp.isfile(tar_file):
                continue
            archive = tarfile.open(tar_file, "r:gz")
            archives[speaker] = archive
            pkl_files[speaker] = sorted([member.name for member in archive.getmembers() if member.name.endswith('.pkl')])
        """ Example:
            self.pkl_files = {
                'chemistry': [
                    'chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg.mp4/68891-00_01_40-00_01_46/68891-00_01_40-00_01_46.pkl',
                    'chemistry/2nd_Order_Rate_Laws-6BZb96mqmbg.mp4/68892-00_01_46-00_01_55/68892-00_01_46-00_01_55.pkl',
                    ...,
                ],
                'conan': [
                    'conan/A_Cameraman_Gets_Too_Close_To_Conan_-_CONAN_on_TBS-5M7NU23GHO4.mkv/114304-00_01_19-00_01_29/114304-00_01_19-00_01_29.pkl',
                    'conan/A_Cameraman_Gets_Too_Close_To_Conan_-_CONAN_on_TBS-5M7NU23GHO4.mkv/114304-00_01_29-00_01_39/114304-00_01_29-00_01_39.pkl',
                    ...,
                ],
            }
        """
        self.archives = archives
        self.pkl_files = pkl_files

    def _load_file(self, filepath: str, speaker: str):
        f = self.archives[speaker].extractfile(filepath)
        dat = pkl.load(f, encoding='latin1')

        body_pose_axis = dat['body_pose_axis']    # (bs, 21, 3)
        expression = dat['expression']            # (bs, 100)
        jaw_pose = dat['jaw_pose']                # (bs, 3)
        betas = dat['betas']                      # (300)
        global_orient = dat['global_orient']      # (bs, 3)
        transl = dat['transl']                    # (bs, 3)
        left_hand_pose = dat['left_hand_pose']    # (bs, 12)
        right_hand_pose = dat['right_hand_pose']  # (bs, 12)
        leye_pose = dat['leye_pose']              # (bs, 3)
        reye_pose = dat['reye_pose']              # (bs, 3)
        
        bs = body_pose_axis.shape[0]

        assert global_orient.ndim == 3 and global_orient.shape[1] == 1, global_orient.shape
        global_orient = global_orient[:, 0, :]

        global_orient[:, :] = global_orient[0, :]
        transl[:, :] = transl[0, :]

        if speaker in ("oliver", "seth", "chemistry"):
            pose_type = 'sitting'
        else:
            pose_type = 'standing'

        if pose_type == 'standing':
            ref_pose = np.zeros(55 * 3)
        elif pose_type == 'sitting':
            ref_pose = np.array([
                0.0, 0.0, 0.0, -1.1826512813568115, 0.23866955935955048, 0.15146760642528534, -1.2604516744613647,
                -0.3160211145877838, -0.1603458970785141, 0.0, 0.0, 0.0, 1.1654603481292725, 0.0, 0.0,
                1.2521806955337524, 0.041598282754421234, -0.06312154978513718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0])
        else:
            assert 0, f'Invalid ref_pose: {ref_pose}'

        body_pose = body_pose_axis.reshape(bs, 63)
        for i in [1, 2, 4, 5, 7, 8, 10, 11]:
            body_pose[:, (i - 1) * 3 + 0] = ref_pose[(i) * 3 + 0]
            body_pose[:, (i - 1) * 3 + 1] = ref_pose[(i) * 3 + 1]
            body_pose[:, (i - 1) * 3 + 2] = ref_pose[(i) * 3 + 2]

        motion_params = {
            'body_pose': body_pose[np.newaxis, :, :],
            'expression': expression[np.newaxis, :, :],
            'jaw_pose': jaw_pose[np.newaxis, :, :],
            'betas': np.broadcast_to(betas, (bs, 300))[np.newaxis, :, :],
            'global_orient': global_orient[np.newaxis, :, :],
            'transl': transl[np.newaxis, :, :],
            'left_hand_pose': left_hand_pose[np.newaxis, :, :],
            'right_hand_pose': right_hand_pose[np.newaxis, :, :],
            'leye_pose': leye_pose[np.newaxis, :, :],
            'reye_pose': reye_pose[np.newaxis, :, :],
        }
        
        return motion_params

    def _load_fast_demo(self, demo_path='./assets/talkshow_demo.npy'):
        smplx_params = np.load(demo_path)
        return {
            'jaw_pose': smplx_params[np.newaxis, :, 0:3],               # 3
            'global_orient': smplx_params[np.newaxis, :, 9:12],         # 3
            'body_pose': smplx_params[np.newaxis, :, 12:75],            # 63
            'left_hand_pose': smplx_params[np.newaxis, :, 75:120],      # 45
            'right_hand_pose': smplx_params[np.newaxis, :, 120:165],    # 45
            'expression': smplx_params[np.newaxis, :, 165:265],         # 100
        }

    def get_smpl_params(self, filename:str, model_type:str='smplx', fps:Optional[int]=None, raw_fps:Optional[int]=None):
        if model_type != 'smplx':
            raise NotImplementedError(f'Invalid model_type for TalkShow: {model_type}')

        if filename == 'demo':
            dat = self._load_fast_demo()
        else:
            # Lazy initialization
            if self.archives is None and self.pkl_files is None:
                self.initialize()
            speaker, fileidx = filename.split('/', maxsplit=1)  # e.g., filename = "chemistry/0"
            filepath = self.pkl_files[speaker][int(fileidx)]
            dat = self._load_file(filepath, speaker)

        num_persons, num_frames, _ = dat['body_pose'].shape

        if fps is not None and raw_fps is not None:
            fps_step = np.ceil(fps / raw_fps)
            slected_frames = [i for i in range(num_frames) if i % fps_step == 0]
            for k in dat.keys():
                dat[k] = dat[k][:, slected_frames, :]

        return dat


if __name__ == '__main__':
    talkshow = TalkShow()

    for k, v in talkshow.pkl_files.items():
        print(f'{k}: {len(v)}')
    
    # chemistry: 2976
    # conan: 6154
    
    dat = talkshow.get_smpl_params("chemistry/0")

    for k, v in dat.items():
        print(k, v.shape)
