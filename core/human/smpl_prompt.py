import os
import os.path as osp
import imageio
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from copy import deepcopy
from typing import Iterable, Union, Optional, List, Any
from numbers import Number
from smplx.body_models import SMPLXOutput
from dataclasses import asdict

from configs import PromptConfig
from data.camera.utils import cyclical_camera
from data.human import load_smpl_sequences
from .smpl_model import SMPLModel, SemanticSMPLModel
from .smpl_condition import SMPL2Condition


def parse_scene_type(scene: str):
    if scene.startswith('canonical'):
        return 'canonical'
    elif scene.startswith('random') or scene == 'vposer':
        return 'random'
    else:
        return 'motion'


def parse_betas(betas: Optional[str], num_betas: int, device: torch.device):
    if not isinstance(betas, str):
        return betas

    betas = eval(betas)
    if isinstance(betas[0], Number):
        betas = (betas,)
    
    betas_list = []
    for b in betas:
        b = torch.tensor(b, device=device, dtype=torch.float32)
        assert b.ndim == 1
        if len(b) < num_betas:
            b = F.pad(b, pad=(0, num_betas - len(b)), mode='constant', value=0.0)
        betas_list.append(b)
    betas = torch.stack(betas_list, dim=0)  # (N, num_betas)

    return betas


def sample_betas(betas: torch.Tensor, i: int = None, max_iteration: int = 25):
    assert betas.ndim == 2 and len(betas) in (1, 2)

    if len(betas) == 1 or i is None:
        return betas[[0]]
    
    r = min(i / max_iteration, 1.0)
    return betas[[0]] * (1 - r) + betas[[1]] * r


class SMPLPrompt:

    def __init__(self, cfg: PromptConfig, smpl_model: SemanticSMPLModel, cond_type: Union[Iterable, str], height: int, width: int, _dataset: Any = None):
        # Init Configs
        self.width = width
        self.height = height
        self.cond_type = cond_type

        self.device = smpl_model.device
        self.model_type = smpl_model.model_type
        self.batch_size = smpl_model.batch_size
        self.scene = cfg.scene
        self.scene_type = parse_scene_type(self.scene)
        self.canonical_pose = cfg.canonical_pose
        self.canonical_mixup_prob = cfg.canonical_mixup_prob
        
        self.num_frame = 1
        self.num_person = cfg.num_person
        
        self.training_ratio: float = 0.0

        # Init Modules
        self.smpl_model = smpl_model
        self.smpl_to_condition = SMPL2Condition(cfg)

        num_betas = smpl_model.model.num_betas
        self.canonical_betas = parse_betas(cfg.canonical_betas, num_betas=num_betas, device=self.device)
        self.observed_betas = parse_betas(cfg.observed_betas, num_betas=num_betas, device=self.device)
        self.max_beta_iteration = cfg.max_beta_iteration

        # Canonical Inputs
        self.centralize_pelvis = cfg.centralize_pelvis
        self.smpl_canonical_inputs = smpl_model.get_smpl_inputs(
            pose_type=self.canonical_pose,
            betas=self.canonical_betas,
            centralize_pelvis=True,  # always True for canonical pose
        )
        self.smpl_canonical_outputs = smpl_model(**self.smpl_canonical_inputs)

        # Observed Inputs
        if self.scene_type == 'canonical':
            self.dynamic = False
            self.smpl_inputs = self.smpl_canonical_inputs
            self.smpl_outputs = self.smpl_canonical_outputs

        elif self.scene_type == 'random':
            self.dynamic = True
            self.smpl_inputs = None
            self.smpl_outputs = None

        else:  # Load SMPL Sequences
            self.dynamic = True
            self.smpl_inputs = None
            self.smpl_outputs = None
            self.camera_sequences = {}
            self.smpl_sequences, self.num_person, self.num_frame = load_smpl_sequences(
                scene=self.scene,
                model_type=self.model_type,
                device=self.device,
                num_person=self.num_person,
                pop_betas=cfg.pop_betas,
                pop_transl=cfg.pop_transl,
                normalize_transl=cfg.normalize_transl,
                centralize_pelvis=cfg.centralize_pelvis,
                pop_global_orient=cfg.pop_global_orient,
                smpl_model=smpl_model,
                frame_interval=cfg.frame_interval,
                camera_sequences=self.camera_sequences,
                _dataset=_dataset,
            )
            if len(self.camera_sequences) == 0:
                self.camera_sequences: Optional[dict] = None

    def get_smpl_inputs_from_sequences(self, frame_idx: Optional[int] = None):
        # Select Frame Index
        if frame_idx is None:
            frame_idx = np.random.randint(0, self.num_frame)
        frame_idx %= self.num_frame

        # Extract SMPL Inputs
        smpl_inputs = {}
        for key, tensor in self.smpl_sequences.items():
            if tensor.ndim in (3, 4, 5):
                smpl_inputs[key] = tensor[:, frame_idx, ...]
            elif tensor.ndim == 2:
                smpl_inputs[key] = tensor
            else:
                assert 0, (key, tensor)

        return smpl_inputs

    def get_camera_params_from_sequences(
        self,
        frame_idx: Optional[int] = None,
        z_near: float = 0.01,
        z_far: float = 1000.0,
        **kwargs,
    ):
        from data.camera.utils import to_projection

        # Select Frame Index
        if frame_idx is None:
            frame_idx = np.random.randint(0, self.num_frame)
        frame_idx %= self.num_frame

        # Check Camera Sequences
        # if self.camera_sequences is None:
        #     return self.get_camera_params(p=frame_idx/self.num_frame, **kwargs)

        # Extract Camera Params
        for k in self.camera_sequences.keys():
            v = self.camera_sequences[k]
            if isinstance(v, np.ndarray):
                self.camera_sequences[k] = torch.tensor(v, device=self.device, dtype=torch.float32)
        
        extrinsic = self.camera_sequences['extrinsic'][frame_idx:frame_idx+1, :, :]
        intrinsics = self.camera_sequences['intrinsics'][frame_idx:frame_idx+1, :, :]
        image_height = self.camera_sequences['image_height']
        image_width = self.camera_sequences['image_width']

        aspect_ratio = self.camera_sequences['aspect_ratio']

        fov_x = self.camera_sequences['fov_x'][frame_idx:frame_idx+1]
        fov_y = self.camera_sequences['fov_y'][frame_idx:frame_idx+1]
        fov = fov_y

        tanfov_x = self.camera_sequences['tanfov_x'][frame_idx:frame_idx+1]
        tanfov_y = self.camera_sequences['tanfov_y'][frame_idx:frame_idx+1]
        tanfov = tanfov_y

        c2w = torch.inverse(extrinsic)

        # Convert to Projection
        projection = to_projection(
            tanfov=tanfov,
            tanfov_x=tanfov_x,
            z_far=z_far,
            z_near=z_near,
        )
        mvp = torch.bmm(projection, extrinsic)

        return {
            'extrinsic': extrinsic,
            'c2w': c2w,
            'intrinsics': intrinsics,
            'mvp': mvp,
            'projection': projection,
            'z_far': z_far,
            'z_near': z_near,
            'fov': fov,
            'fov_x': fov_x,
            'fov_y': fov_y,
            'tanfov': tanfov,
            'tanfov_y': tanfov_y,
            'tanfov_x': tanfov_x,
            'aspect_ratio': aspect_ratio,
            'image_height': image_height,
            'image_width': image_width,
        }

    def get_camera_params(self, p, **kwargs):
        return cyclical_camera(
            p=p,
            device=self.device,
            image_height=self.height,
            image_width=self.width,
            **kwargs,
        )

    def get_cond_images(
        self,
        smpl_outputs: SMPLXOutput,
        camera_params: dict,
        cond_type=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> List[Image.Image]:
        """
            Args:
              - smpl_outputs: SMPLXOutput
              - camera_params: dict
              - cond_type: List[str]
        """
        # Prepare Inputs
        if cond_type is None:
            cond_type = self.cond_type
        if isinstance(cond_type, str):
            cond_type = [cond_type,]
        condition_height = height if height is not None else self.height
        condition_width = width if width is not None else self.width
        # Export Condition Images
        cond_images = []
        for cond in cond_type:
            cond_image = self.smpl_to_condition(
                smpl_outputs=smpl_outputs,
                triangles=self.triangles,
                camera_params=camera_params,
                condition_type=cond,
                condition_height=condition_height,
                condition_width=condition_width,
            )
            cond_images.append(cond_image)
        # Return
        return cond_images

    def __call__(self, frame_idx: Optional[int] = None, batch_idx: Optional[int] = None):

        extra_smpl_inputs = {}
        if self.observed_betas is not None:
            betas = sample_betas(self.observed_betas, i=batch_idx, max_iteration=self.max_beta_iteration)
            extra_smpl_inputs['betas'] = betas
        
        if self.scene_type == 'canonical':
            if self.scene == 'canonical' or self.scene == self.canonical_pose:
                if len(extra_smpl_inputs) > 0:
                    smpl_inputs = extra_smpl_inputs
                    smpl_inputs.update(self.smpl_inputs)
                    smpl_outputs = self.smpl_model(**smpl_inputs)
                else:
                    smpl_inputs = self.smpl_inputs
                    smpl_outputs = self.smpl_outputs
            else:
                if self.scene in ('canonical-loop', 'canonical-loop2'):
                    smpl_inputs = self.smpl_model.get_smpl_inputs(
                        pose_type=self.scene,
                        training_ratio=self.training_ratio,
                        centralize_pelvis=True,
                    )
                else:
                    smpl_inputs = self.smpl_model.get_smpl_inputs(
                        pose_type=self.scene,
                        centralize_pelvis=True,
                    )
                smpl_inputs.update(extra_smpl_inputs)
                smpl_outputs = self.smpl_model(**smpl_inputs)
            
        elif self.scene_type == 'random':
            smpl_inputs = self.smpl_model.get_smpl_inputs(
                pose_type=self.scene,
                centralize_pelvis=True,
                canonical_mixup_prob=self.canonical_mixup_prob,
            )
            smpl_inputs.update(extra_smpl_inputs)
            smpl_outputs = self.smpl_model(**smpl_inputs)
        
        else:
            ###########################################################################
            if self.observed_betas is not None and len(self.observed_betas) > 1:
                frame_idx = max(self.max_beta_iteration, frame_idx)
            ###########################################################################
            smpl_inputs = self.get_smpl_inputs_from_sequences(frame_idx)
            smpl_inputs.update(extra_smpl_inputs)
            if self.num_person == self.batch_size:
                smpl_outputs = self.smpl_model(**smpl_inputs)
            else:
                smpl_outputs = None
                for i in range(self.num_person):
                    _smpl_inputs = {}
                    for k, v in smpl_inputs.items():
                        _smpl_inputs[k] = v[i:i+1, ...]
                    _smpl_outputs = self.smpl_model(**_smpl_inputs)
                    if smpl_outputs is None:
                        smpl_outputs = asdict(_smpl_outputs)
                    else:
                        for k in smpl_outputs.keys():
                            if smpl_outputs[k] is not None:
                                smpl_outputs[k] = torch.cat((smpl_outputs[k], _smpl_outputs[k]), dim=0)
                smpl_outputs = SMPLXOutput(**smpl_outputs)
    
        return smpl_inputs, smpl_outputs

    @property
    def triangles(self) -> np.ndarray:
        return self.smpl_model.model.faces

    def write_video(
        self,
        save_dir: str,
        cond_type: Optional[str] = None,
        eval_size: Optional[int] = None,
        save_sequence: bool = False,
        save_one_image: bool = False,
        trajectory: str = 'circle',
        video_name: Optional[str] = None,
        **kwargs,
    ):
        from utils.video import VideoWriterPyAV as VideoWriter
        # init
        if video_name is None:
            video_name = f'{cond_type}.mp4'
        else:
            video_name = f'{video_name}_{cond_type}.mp4'
        video_writer = VideoWriter(video_path=osp.join(save_dir, video_name), fps=25)
        four_views = []
        os.makedirs(save_dir, exist_ok=True)
        # eval size
        if cond_type is None:
            cond_type = self.cond_type[0] if isinstance(self.cond_type, Iterable) else self.cond_type
        if eval_size is None:
            eval_size = 100 if self.num_frame == 1 else self.num_frame
        # run
        for i in range(eval_size):
            camera_params = cyclical_camera(
                p=i/eval_size,
                device=self.device,
                image_height=self.height,
                image_width=self.width,
                trajectory=trajectory,
                **kwargs,
            )
            _, smpl_outputs = self.__call__(frame_idx=i)
            image = self.get_cond_images(
                smpl_outputs=smpl_outputs,
                camera_params=camera_params,
                cond_type=cond_type,
            )[0]
            if save_one_image and i == 0:
                image.save(osp.join(save_dir, f'{cond_type}.png'))
            if i in (0, 25, 50, 75):
                four_views.append(image)
            if save_sequence:
                image.save(osp.join(save_dir, f'{cond_type}_{i:04d}.jpg'))
            video_writer.write(image)
        video_writer.release()
        return four_views
