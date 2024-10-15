import numpy as np
import os
import os.path as osp
import pickle as pkl
from configs.paths import AIST_ROOT


class AIST(object):
    def __init__(self, root: str = AIST_ROOT, from_zip=True) -> None:
        dat = {}
        if from_zip:
            zip_file = osp.join(root, '20210308_motions.zip')
            assert osp.isfile(zip_file), f'{zip_file} not found'
            import zipfile
            archive = zipfile.ZipFile(zip_file, 'r')
            for filepath in archive.namelist():
                if not filepath.endswith('.pkl'):
                    continue
                filename = osp.splitext(filepath.split('/')[-1])[0]
                dat[filename] = filepath
            self.archive = archive
        else:
            for filename in sorted(os.listdir(root)):
                if not filename.endswith('.pkl'):
                    continue
                filepath = osp.join(root, filename)
                filename = osp.splitext(filename)[0]
                dat[filename] = filepath
        self.dat = dat
        self.from_zip = from_zip

    def get_smpl_params(self, filename, model_type='smpl', fps=60, stand_fps=25, raw_data=False):

        if self.from_zip:
            dat = pkl.load(self.archive.open(self.dat[filename], "r"))
        else:
            dat = pkl.load(open(self.dat[filename], 'rb'))
        # dat.keys(): ['smpl_loss', 'smpl_poses', 'smpl_scaling', 'smpl_trans']

        if raw_data:
            return dat

        pose_params = dat['smpl_poses'][np.newaxis, ...]  # [1, F, 24*3]
        global_orient = pose_params[:, :, :3]
        body_pose = pose_params[:, :, 3:]

        transl = dat['smpl_trans'][np.newaxis, ...] / dat['smpl_scaling']  # [1, F, 3]

        fps_step = np.ceil(fps / stand_fps)
        slected_frames = [i for i in range(pose_params.shape[1]) if i % fps_step == 0]
        global_orient = global_orient[:, slected_frames, :]
        body_pose = body_pose[:, slected_frames, :]
        transl = transl[:, slected_frames, :]

        if model_type in ('smplx', 'smplh'):
            # Layout of 23 joints of SMPL
            # https://www.researchgate.net/figure/Layout-of-23-joints-in-the-SMPL-models_fig2_351179264
            body_pose = body_pose[:, :, :-2*3]

        return {
            'global_orient': global_orient,  # (2, F, 3)
            'body_pose': body_pose,          # (2, F, 63/69)
            'transl': transl,                # (2, F, 3)
        }
