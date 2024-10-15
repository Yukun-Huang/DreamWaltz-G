import os
import os.path as osp
import torch
import numpy as np
from configs.paths import AMASS_ROOT


class AMASS(object):
    def __init__(self, root: str = AMASS_ROOT) -> None:
        self.root = root
        filenames = os.listdir(root)
        filepaths = [osp.join(root, fn) for fn in filenames]
        dat = {}
        for fn, fp in zip(filenames, filepaths):
            fn = osp.splitext(fn)[0]
            dat[fn] = fp
        self.dat = dat

    def get_smpl_params(self, filename, model_type='smpl'):
        bdata = np.load(self.dat[filename], allow_pickle=True)
        # FPS
        try:
            fps = bdata['mocap_framerate']
            frame_number = bdata['trans'].shape[0]
        except Exception as e:
            fps = 100
            print(e)
        stand_fps = 25
        fps_step = np.ceil(fps / stand_fps)
        # fps_clamp = (100, frame_number - 100)
        # slected_frames = [i for i in range(frame_number) if i % fps_step == 0 and fps_clamp[0] <= i <= fps_clamp[1]]
        slected_frames = [i for i in range(frame_number) if i % fps_step == 0]

        # Parse
        global_orient = bdata['poses'][:, :3][np.newaxis, ...]      # controls the global root orientation
        body_pose = bdata['poses'][:, 3:(21*3+3)][np.newaxis, ...]  # controls the body
        hand_pose = bdata['poses'][:, 66:][np.newaxis, ...]         # controls the finger articulation
        betas = bdata['betas'][:10][np.newaxis] * 0.0    # controls the body shape
        transl = bdata['trans'][np.newaxis, ...]
    
        # Preprocess
        global_orient = global_orient[:, slected_frames, :]
        body_pose = body_pose[:, slected_frames, :]
        hand_pose = hand_pose[:, slected_frames, :]
        transl = transl[:, slected_frames, :]

        # Preprocess
        # global_orient[:, :, 0], global_orient[:, :, 2] = global_orient[:, :, 2], global_orient[:, :, 0]
        # global_orient -= global_orient[:, 0:1, :]
        global_orient = np.zeros_like(global_orient)

        if model_type in ('smpl', ):
            # Layout of 23 joints of SMPL
            # https://www.researchgate.net/figure/Layout-of-23-joints-in-the-SMPL-models_fig2_351179264
            body_pose = np.concatenate((body_pose, np.zeros_like(body_pose)[..., :6]), axis=-1)

        smpl_params = {
            'global_orient': global_orient,  # (2, F, 3)
            'body_pose': body_pose,          # (2, F, 63/69)
            'betas': betas,                  # (2, F, 10)
            'transl': transl,                # (2, F, 3)
        }

        return smpl_params
