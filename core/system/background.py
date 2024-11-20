import torch
import torch.nn as nn
import numpy as np
import os
import os.path as osp
import cv2
from typing import Optional
from configs import TrainConfig
from core.nerf.encoding import get_encoder
from core.nerf.nerf_utils import get_rays
from core.nerf.nerf_model import MLP


class PureColorBackground:
    COLOR_MAPPING = {
        'black': torch.tensor([0.0, 0.0, 0.0]),
        'white': torch.tensor([1.0, 1.0, 1.0]),
        'gray': torch.tensor([0.5, 0.5, 0.5]),
    }

    COLOR_VALUE_MAPPING = {
        'black': 0.0,
        'white': 1.0,
        'gray': 0.5,
    }

    def __init__(self) -> None:
        pass

    def __contains__(self, bg_mode: str):
        return bg_mode in ['black', 'white', 'gray']
    
    @staticmethod
    def get_background(
        color: str,
        image_width: int,
        image_height: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        bg_color = PureColorBackground.COLOR_MAPPING[color]
        bg_image = bg_color[None, None, None, :].repeat(batch_size, image_height, image_width, 1)
        return bg_image.to(device)
    
    @staticmethod
    def get_background_like(
        color: str,
        image: torch.Tensor,
    ) -> torch.Tensor:
        bg_color_value = PureColorBackground.COLOR_VALUE_MAPPING[color]
        bg_image = torch.ones_like(image) * bg_color_value
        return bg_image


class MLPBackground(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_bg, in_dim_bg = get_encoder('frequency_torch', multires=4)
        self.bg_net = MLP(dim_in=in_dim_bg, dim_out=3, dim_hidden=16, num_layers=2)

    def forward(self, data, skip_bg=False):
        """
        Args:
            c2w: torch.Tensor, [B, 4, 4]
            intrinsics: torch.Tensor, [B, 3, 3]
        Return:
            image: torch.Tensor, [B, H, W, 3]
        """
        c2w = data['c2w']
        intrinsics = data['intrinsics']
        H = data['image_height']
        W = data['image_width']

        batch_size, device = c2w.shape[0], c2w.device

        if not skip_bg:
            rays_d = get_rays(c2w=c2w, intrinsics=intrinsics, H=H, W=W)['rays_d']
            dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            image = self.bg_net(self.encoder_bg(dirs.view(-1, 3)))
            image = torch.sigmoid(image).view(batch_size, H, W, 3).contiguous()
        else:
            image = torch.zeros(batch_size, H, W, 3).to(device)

        return image

    def get_optimizer(self, cfg:TrainConfig=None):
        from core.optim.adan import Adan
        optimizer = Adan(self.parameters(), lr=0.001, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        return optimizer


class VideoBackground:
    def __init__(self, path: str, preload: bool = True) -> None:
        # Load Video
        if osp.isfile(path):
            video_path = path
            self.use_temp_cache = False
        elif path.startswith('motionx_reenact,'):
            import tempfile
            from data.human.motionx_reenact import MotionX_ReEnact
            motionx_reenact = MotionX_ReEnact()
            video_path = osp.join(tempfile.gettempdir(), f'{path}.mp4')
            filename = path.replace('motionx_reenact,', '', 1)
            motionx_reenact.extract_video(filename, save_path=video_path, video_type='inpainting')
            self.use_temp_cache = True
        else:
            raise FileNotFoundError(f"Video File Not Found: {path}")
        # Set Properties
        frame_source = cv2.VideoCapture(video_path)
        self.fps = int(frame_source.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(frame_source.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(frame_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(frame_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_path = video_path
        self.frame_source = frame_source
        # Preload
        self.frame_cache = self.load_all_frames() if preload else None

    def load_all_frames(self):
        frames = []
        for i in range(self.frame_count):
            ret, frame = self.frame_source.read()
            if ret:
                frames.append(frame)
            else:
                raise ValueError(f"Failed to Read Frame at Index: {i}")
        return frames

    def get_background(self, frame_index: int) -> np.ndarray:
        if self.frame_cache is not None:
            return self.frame_cache[frame_index]
        else:
            self.frame_source.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.frame_source.read()
            if ret:
                return frame
            else:
                raise ValueError(f"Failed to Read Frame at Index: {frame_index}")

    def get_background_like(self, frame_index: int, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_index: int
            image: torch.Tensor, [B, H, W, 3]
        Return:
            bg_image: torch.Tensor, [B, H, W, 3]
        """
        frame = self.get_background(frame_index)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_width, image_height = image.shape[2], image.shape[1]
        if frame.shape[1] != image_width or frame.shape[0] != image_height:
            frame = cv2.resize(frame, (image_width, image_height))
        bg_image = torch.from_numpy(frame).float() / 255.0
        return bg_image.to(image)

    def __del__(self):
        self.frame_source.release()
        self.frame_cache = None
        if self.use_temp_cache:
            os.remove(self.video_path)
