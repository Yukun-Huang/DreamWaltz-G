# import pickle as pkl
from typing import Any
import numpy as np
import os
import os.path as osp
import pickle as pkl
from glob import glob


abbreviation = {
    'dance': 'courtyard_dancing_00',
    'basketball': 'courtyard_basketball_00',
    'capoeira': 'courtyard_capoeira_00',
    'warmwelcome': 'courtyard_warmWelcome_00',
    'selfies': 'courtyard_captureSelfies_00',
    'arguing': 'courtyard_arguing_00',
    'jumpbench': 'courtyard_jumpBench_01',
}


# 3DPW Dataset
class PW3D:
    def __init__(self, root, from_zip=True) -> None:
        root = osp.join(root, "3DPW")
        dat = {}
        if from_zip:
            seq_zip_file = osp.join(root, 'sequenceFiles.zip')
            smplx_zip_file = osp.join(root, 'SMPL-X.zip')
            import zipfile
            seq_archive = zipfile.ZipFile(seq_zip_file, 'r')
            smplx_archive = zipfile.ZipFile(smplx_zip_file, 'r')
            for filepath in seq_archive.namelist():
                if not filepath.endswith('.pkl') or '__MACOSX' in filepath:
                    continue
                filename = osp.splitext(filepath.split('/')[-1])[0]
                dat[filename] = filepath
            self.seq_archive = seq_archive
            self.smplx_archive = smplx_archive
        else:
            seq_dir = osp.join(root, 'sequenceFiles')
            for subset in ('train', 'validation', 'test'):
                for seq_path in glob(osp.join(seq_dir, subset, '*.pkl')):
                    scene_name = osp.splitext(osp.split(seq_path)[-1])[0]
                    assert scene_name not in dat.keys()
                    dat[scene_name] = seq_path
        self.dat = dat
        self.from_zip = from_zip

    def get_smpl_params(self, filename, model_type='smpl'):
        # Get the SMPL body parameters - poses, betas and translation parameters
        # https://github.com/aymenmir1/3dpw-eval/blob/master/evaluate.py
        if filename in abbreviation:
            filename = abbreviation[filename]

        if self.from_zip:
            data_buffer = self.seq_archive.open(self.dat[filename], 'r')
            data = pkl.load(data_buffer, encoding='latin1')
        else:
            data = pkl.load(open(self.dat[filename], 'rb'), encoding='latin1')
        """
        For each sequence, the .pkl-file contains a dictionary with the following fields:
        - sequence: String containing the sequence name
        - betas: SMPL shape parameters for each actor which has been used for tracking (List of 10x1 SMPL beta parameters)
        - poses: SMPL body poses for each actor aligned with image data (List of Nx72 SMPL joint angles, N = #frames)
        - trans: tranlations for each actor aligned with image data (List of Nx3 root translations)
        - poses_60Hz: SMPL body poses for each actor at 60Hz (List of Nx72 SMPL joint angles, N = #frames)
        - trans_60Hz: tranlations for each actor at 60Hz (List of Nx3 root translations)
        - betas_clothed: SMPL shape parameters for each clothed actor (List of 10x1 SMPL beta parameters)
        - v_template_clothed: 
        - gender: actor genders (List of strings, either 'm' or 'f')
        - texture_maps: texture maps for each actor
        - poses2D: 2D joint detections in Coco-Format for each actor (only provided if at least 6 joints were detected correctly)
        - jointPositions: 3D joint positions of each actor (List of Nx(24*3) XYZ coordinates of each SMPL joint)
        - img_frame_ids: an index-array to down-sample 60 Hz 3D poses to corresponding image frame ids
        - cam_poses: camera extrinsics for each image frame (Ix4x4 array, I frames times 4x4 homegenous rigid body motion matrices)
        - campose_valid: a boolean index array indicating which camera pose has been aligned to the image
        - cam_intrinsics: camera intrinsics (K = [f_x 0 c_x;0 f_y c_y; 0 0 1])
        """

        betas = np.array(data['betas'])  # [Np, Nf, 10]
        transl = np.array(data['trans'])  # [Np, Nf, 3]

        pose_params = np.array(data['poses'])   # [Np, Nf, 24*3]
        global_orient = pose_params[:, :, :3]
        body_pose = pose_params[:, :, 3:]

        smpl_params = {
            'global_orient': global_orient,  # (2, F, 3)
            'body_pose': body_pose,          # (2, F, 63/69)
            'betas': betas,                  # (2, F, 10)
            'transl': transl,                # (2, F, 3)
        }

        if model_type in ('smplx', 'smplh'):
            # Layout of 23 joints of SMPL
            # https://www.researchgate.net/figure/Layout-of-23-joints-in-the-SMPL-models_fig2_351179264
            smpl_params['body_pose'] = smpl_params['body_pose'][:, :, :-2*3]
            smpl_params.pop('betas')

        return smpl_params
