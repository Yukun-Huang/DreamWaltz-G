import random
from typing import Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from smplx.utils import SMPLXOutput

from configs import DataConfig
from .utils import RandomCamera, RandomCamera4Avatar, CyclicalCamera, CyclicalCamera4Avatar

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.human.smpl_prompt import SMPLPrompt


class CameraDataset:
    """ Return camera pose and view-dependent text prompt. """
    def __init__(
            self,
            cfg: DataConfig,
            device: torch.device,
            mode: str,
            H: int = 512,
            W: int = 512,
            eval_size: int = 100,
        ):
        assert mode in ('train', 'val', 'test')

        self.cfg = cfg
        self.device = device
        self.training = mode in ['train',]
        self.eval_size = eval_size
        self.training_ratio: float = None

        self.camera = self.build_camera(cfg=cfg, device=device, H=H, W=W)

    def build_camera(self, cfg: DataConfig, device: torch.device, H: int = 512, W: int = 512):
        if self.cfg.cameras is None:
            if self.training:
                return RandomCamera(device=device, cfg=cfg, image_height=H, image_width=W)
            else:
                return CyclicalCamera(device=device, cfg=cfg, image_height=H, image_width=W)
        else:
            raise NotImplementedError

    def set_training_ratio(self, train_step, max_iteration):
        self.training_ratio = train_step / max_iteration
        if self.cfg.progressive_radius:
            self.camera.training_ratio = self.training_ratio

    def collate(self, index: list) -> dict:
        if self.training:
            size = len(index)
            return self.camera(size=size)
        else:
            p = index[0] / self.eval_size
            return self.camera(p=p)

    def dataloader(self, batch_size=1):
        loader = DataLoader(list(range(self.eval_size)), batch_size=batch_size, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader


class CameraDatasetWithSMPL(CameraDataset):
    """
    Return camera pose, view-dependent text prompt, and view-dependent smpl prompt.
    """
    def __init__(
        self,
        cfg: DataConfig,
        device: torch.device,
        smpl_prompt: 'SMPLPrompt',
        H: int = 512,
        W: int = 512,
        **kwargs,
    ):
        super().__init__(cfg=cfg, device=device, H=H, W=W, **kwargs)
        self.smpl_prompt = smpl_prompt
        # For Animation
        self.fix_animation = cfg.eval_fix_animation
        if self.smpl_prompt.num_frame > 1:
            self.num_frame = self.smpl_prompt.num_frame
            if not self.training and not self.fix_animation and self.num_frame > 0:
                self.eval_size = self.num_frame
        else:
            self.num_frame = None
        # SMPL-guided Random Camera Sampling
        if not smpl_prompt.dynamic:
            self.setup_camera_offset(smpl_prompt.smpl_outputs, verbose=True)
        # Update random pose per K iterations
        if self.training:
            self.count = 0
            self.buffer = {}
            self.random_pose_iter = cfg.random_pose_iter

    def set_training_ratio(self, train_step, max_iteration):
        super().set_training_ratio(train_step, max_iteration)
        self.smpl_prompt.training_ratio = self.training_ratio

    def build_camera(self, cfg: DataConfig, device: torch.device, H: int = 512, W: int = 512):
        if self.training:
            return RandomCamera4Avatar(device=device, cfg=cfg, image_height=H, image_width=W)
        else:
            return CyclicalCamera4Avatar(device=device, cfg=cfg, image_height=H, image_width=W)

    def get_frame_index(self, index, skip_frame=False):
        if self.training or self.num_frame is None:
            frame_idx = None  # uniform sampling for training
        else:
            assert len(index) == 1
            if self.fix_animation:
                frame_idx = 0
            elif skip_frame:
                frame_idx = int((index[0] / self.eval_size) * self.num_frame)
            else:
                frame_idx = index[0]
        return frame_idx

    def setup_camera_offset(self, smpl_outputs: SMPLXOutput, verbose: bool = False):
        joints = smpl_outputs.joints.cpu().numpy()
        self.camera.setup_camera_offset(joints, verbose=verbose)

    def collate(self, index) -> dict:
        # smpl forward
        frame_idx = self.get_frame_index(index)

        # training mode
        if self.training:
            if self.random_pose_iter > 0:
                if self.count % self.random_pose_iter == 0:
                    smpl_inputs, smpl_outputs = self.smpl_prompt(frame_idx=frame_idx)
                    self.buffer['smpl_inputs'] = smpl_inputs
                    self.buffer['smpl_outputs'] = smpl_outputs
                else:
                    smpl_inputs = self.buffer['smpl_inputs']
                    smpl_outputs = self.buffer['smpl_outputs']
                self.count += 1
            else:
                smpl_inputs, smpl_outputs = self.smpl_prompt(frame_idx=frame_idx)
        else:
            smpl_inputs, smpl_outputs = self.smpl_prompt(frame_idx=frame_idx, batch_idx=index[0])

        # sample camera parameters
        if not self.training and self.cfg.eval_camera_track == 'predefined':
            camera_params = self.smpl_prompt.get_camera_params_from_sequences(frame_idx=frame_idx)
        else:
            if self.smpl_prompt.dynamic:
                self.setup_camera_offset(smpl_outputs)
            camera_params = super().collate(index=index)

        # render control image
        cond_images = self.smpl_prompt.get_cond_images(
            smpl_outputs=smpl_outputs,
            camera_params=camera_params,
        )

        # data batch
        data_batch = camera_params
        data_batch['smpl_inputs'] = smpl_inputs
        data_batch['smpl_outputs'] = smpl_outputs
        data_batch['cond_images'] = cond_images
        data_batch['frame_index'] = frame_idx

        return data_batch


if __name__ == '__main__':
    for one_batch in CameraDataset(DataConfig(), torch.device('cpu')).dataloader():
        print(one_batch)
        break
