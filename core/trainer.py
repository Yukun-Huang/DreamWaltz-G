import os.path as osp
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List, Optional
from numbers import Number
import imageio
import bisect
from random import choice, random
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

from configs import TrainConfig

from utils import make_path, seed_everything
from utils.image import tensor2image

from data.camera import CameraDataset, CameraDatasetWithSMPL
from data.iterator import DataLoaderManager

from core.human.smpl_model import SemanticSMPLModel as SMPLModel
from core.human.smpl_prompt import SMPLPrompt
from core.guidance.text import TextAugmentation as ViewPrompt


class _Visualizer:
    train_step: int
    snapshot_train_path: Path
    snapshot_eval_path: Path

    def snapshot_image(
        self,
        image: torch.Tensor,
        filename: str,
        color_channel: int,
        extname: str = 'png',
    ) -> None:
        save_path = self.snapshot_train_path / filename / f'{filename}_{self.train_step:05d}.{extname}'
        save_path.parent.mkdir(exist_ok=True)
        tensor2image(image.detach(), color_channel=color_channel).save(save_path)

    @staticmethod
    def concat_alpha(image: torch.Tensor, alpha: torch.Tensor, color_channel: int = 3):
        assert color_channel == 3
        if image.shape[1:3] != alpha.shape[1:3]:
            alpha = alpha.permute(0, 3, 1, 2).contiguous()
            alpha = F.interpolate(alpha, size=image.shape[1:3], mode='bilinear', align_corners=False)
            alpha = alpha.permute(0, 2, 3, 1).contiguous()
        return torch.cat([image, alpha], dim=3)

    def latent_to_image(self, latent: torch.Tensor, color_channel: int):
        vae_decode = self.diffusion.decode_latents
        if color_channel == 3:
            latent = latent.permute(0, 3, 1, 2).contiguous()
        if latent.shape[-2] > 128 or latent.shape[-1] > 128:
            latent = F.interpolate(latent, (128, 128), mode='area')
        image = vae_decode(latent)
        if color_channel == 3:
            image = image.permute(0, 2, 3, 1).contiguous()
        return image
    
    def snapshot_latent(
        self,
        latent: torch.Tensor,
        filename: str,
        color_channel: int,
    ) -> None:
        image = self.latent_to_image(latent.detach(), color_channel=color_channel)
        self.snapshot_image(image, color_channel=1, filename=filename)

    def snapshot_feature(
        self,
        feature: torch.Tensor,
        filename: str,
        color_channel: int,
        save_raw: bool = False,
        normalize: bool = True,
        normalization_method: str = 'std',
        is_latent: bool = False,
        pseudo_color: bool = False,
    ) -> None:
        feature = feature.detach()

        if save_raw:
            save_path = self.snapshot_train_path / f'{filename}_raw' / f'{filename}_{self.train_step:05d}.pt'
            save_path.parent.mkdir(exist_ok=True)
            torch.save(feature[0].detach().cpu(), save_path)
        
        if color_channel == 3:
            feature = feature.permute(0, 3, 1, 2).contiguous()
        else:
            assert color_channel == 1
        
        if normalize:
            if normalization_method == 'std':
                feature = feature / torch.std(feature)
            elif normalization_method == 'max':
                feature = feature / feature.std().max()
            else:
                feature = feature / torch.norm(feature.flatten(), p=2)

        if is_latent:
            self.snapshot_latent(feature, filename=filename, color_channel=1)
        else:
            image = feature.detach().abs().mean(dim=1, keepdim=True)
            if pseudo_color:
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                cmap = plt.get_cmap('jet')
                plt.gca().imshow(image[0].squeeze(0).cpu().numpy(), cmap=cmap)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
            else:
                self.snapshot_image(image, filename=filename, color_channel=1)

    def snapshot_condition(
        self,
        image: Union[List[Image.Image], Image.Image],
        filename: Union[List[str], str],
        dirname: str = 'conditions',
    ) -> None:
        if isinstance(image, list) and isinstance(filename, List):
            images, filenames = image, filename
            for image, filename in zip(images, filenames):
                self.snapshot_condition(image, filename, dirname)
        else:
            save_path = self.snapshot_train_path / dirname / f'{filename}_{self.train_step:05d}.png'
            save_path.parent.mkdir(exist_ok=True)
            image.save(save_path)

    @torch.inference_mode()
    def snapshot(self, data, render_outputs, sd_outputs):
        for var in ('image', 'image_fg', 'image_gradients'):
            kwargs = {'filename': var, 'color_channel': 3}
            if var in render_outputs:
                if var == 'image_gradients':
                    # grad = (render_outputs[var] / (render_outputs[var].abs().max() + 1e-9)) / 2.0
                    grad = (render_outputs[var] / (render_outputs[var].std() + 1e-9)) / 2.0
                    self.snapshot_image(grad + 0.5, **kwargs)
                    kwargs['filename'] = 'image_targets'
                    self.snapshot_image(render_outputs['image'] + grad, **kwargs)
                elif var.startswith('image'):
                    image = render_outputs[var]
                    if image.size(kwargs['color_channel']) == 4:
                        image = self.latent_to_image(image, color_channel=kwargs['color_channel'])
                    if var == 'image_fg':
                        image = self.concat_alpha(image, alpha=render_outputs['alpha'], color_channel=3)
                    self.snapshot_image(image, **kwargs)
                else:
                    self.snapshot_image(render_outputs[var], **kwargs)

        for var in ('gradients', 'targets', 'latents_noisy', 'latents_1orig', 'latents_final'):  # 'latents_1step'
            if var in sd_outputs:
                kwargs = {'filename': var, 'color_channel': 1}
                if var == 'gradients':
                    self.snapshot_feature(sd_outputs[var], **kwargs, is_latent=True)  # save_raw=self.train_step in (1, 100, 300, 500)
                else:
                    self.snapshot_latent(sd_outputs[var], **kwargs)

        if 'cond_images' in data:
            self.snapshot_condition(data['cond_images'], filename=self.controlnet_condition)


class _Checkpointer:
    cfg: TrainConfig

    def get_latest_checkpoint(self, ckpt_path):
        # latest ckpt from a file
        if osp.isfile(ckpt_path):
            return ckpt_path
        # latest ckpt from a directory
        if osp.isdir(ckpt_path):
            ckpt_dir = ckpt_path
        else:
            ckpt_dir = osp.join(self.cfg.log.exp_root, ckpt_path)
        checkpoint_list = sorted(Path(ckpt_dir).glob('*.pth'))
        if checkpoint_list:
            checkpoint = checkpoint_list[-1]
            logger.info(f"Latest checkpoint is {checkpoint}")
            return checkpoint
        # ckpt not found
        raise FileNotFoundError(f"No checkpoint file found in {ckpt_path}")
        # logger.info(f"No checkpoint file found in {ckpt_path}")
        # return None

    def load_checkpoint(self, checkpoint, model_only=False):
        if model_only:
            logger.info(f'Load model weights from {checkpoint}...')
        else:
            logger.info(f'Load model weights and optimizer checkpoints from {checkpoint}...')

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            logger.info("Loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("Loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        self.past_checkpoints = checkpoint_dict['checkpoints']

        if self.cfg.optim.resume or self.cfg.log.eval_only:
            self.train_step = checkpoint_dict['train_step'] # + 1
            logger.info(f"Loaded at step {self.train_step}")

        if model_only:
            return

        if hasattr(self, 'optimizers') and self.optimizers is not None and 'optimizers' in checkpoint_dict:
            try:
                for optimizer, state_dict in zip(self.optimizers, checkpoint_dict['optimizer']):
                    optimizer.load_state_dict(state_dict)
                logger.info("Loaded optimizer.")
            except Exception as e:
                logger.warning("Failed to load optimizer.")

        if hasattr(self, 'scaler') and self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("Loaded scaler.")
            except Exception as e:
                logger.warning("Failed to load scaler.")

    def save_checkpoint(self, full=False):
        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if full:
            state['optimizers'] = [optimizer.state_dict() for optimizer in self.optimizers]
            state['scaler'] = self.scaler.state_dict()

        state['model'] = self.model.state_dict()

        file_path = f"step_{self.train_step:06d}.pth"

        if len(self.past_checkpoints) == 0 or file_path != self.past_checkpoints[-1]:
            self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)

    def init_checkpoints(self):
        if self.cfg.optim.ckpt is not None:
            checkpoint = self.get_latest_checkpoint(self.cfg.optim.ckpt)
        elif self.cfg.optim.resume:
            checkpoint = self.get_latest_checkpoint(self.exp_path / 'checkpoints')
        else:
            checkpoint = None

        if checkpoint is not None:
            model_only = self.cfg.log.eval_only or not self.cfg.optim.resume
            self.load_checkpoint(checkpoint, model_only=model_only)
        
        # Extra Checkpoints
        if self.cfg.optim.ckpt_extra is not None:
            checkpoint = self.get_latest_checkpoint(self.cfg.optim.ckpt_extra)
            logger.info(f'Load extra model weights from {checkpoint}...')
            checkpoint_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_extra_avatar_from_state_dict(checkpoint_dict['model'])


class _Logger:
    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

    @property
    def time_to_checkpoint(self) -> bool:
        return self.train_step % self.cfg.log.save_interval == 0 or self.train_step == 1

    @property
    def time_to_evaluate(self) -> bool:
        return self.train_step % self.cfg.log.evaluate_interval == 0 or self.train_step in (1, 250)

    @property
    def time_to_snapshot(self) -> bool:
        return self.train_step % self.cfg.log.snapshot_interval == 0 or self.train_step in (1, 100, 200, 300, 400)


class Trainer(_Visualizer, _Checkpointer, _Logger):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.use_controlnet = self.cfg.guide.use_controlnet
        self.controlnet_condition = self.cfg.guide.controlnet_condition
        # if self.cfg.character is not None:
        #     self.character_description = get_character(self.cfg.character)
        #     self.cfg.guide.text = self.character_description['text_prompt']

        seed_everything(self.cfg.optim.seed)

        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')

        self.init_logger()

        # 0. Init View-dependent Text Prompt
        self.view_prompt = ViewPrompt(text=self.cfg.guide.text, cfg=self.cfg.prompt)

        # 1. Init Diffusion Model
        self.diffusion = self.init_diffusion()
        if hasattr(self.diffusion, 'get_text_embeds'):
            self.text_embeds_dict = self.init_text_embeddings(self.diffusion)
        if hasattr(self.diffusion, 'default_image_size'):
            self.diffusion_image_height = self.diffusion.default_image_size
            self.diffusion_image_width = self.diffusion.default_image_size
        else:
            self.diffusion_image_height = 512
            self.diffusion_image_width = 512

        # 2. Init Human Model
        self.smpl_model, self.smpl_prompt = self.init_human_template()

        # 3. Init 3D Model
        if self.cfg.stage == 'nerf':
            self.model = self.init_nerf_model()
        elif self.cfg.stage == 'gs':
            self.model = self.init_gaussian_model()
        elif self.cfg.stage == 'mesh':
            from core.human.smpl_renderer import SMPLRenderer
            self.smpl_renderer = SMPLRenderer(smpl_model=self.smpl_model, body_mapping=None, albedo='smplx')
        else:
            assert 0, self.cfg.stage

        # 3. Init Data Loaders
        self.dataloaders = self.init_dataloaders()

        # 4. Init checkpoints
        self.past_checkpoints = []
        self.init_checkpoints()

        # 5. Training-only Initialization
        if not self.cfg.log.eval_only:
            # Make path
            self.snapshot_train_path = make_path(self.exp_path / 'snapshots' / 'train')
            self.snapshot_eval_path = make_path(self.exp_path / 'snapshots' / 'eval')

            # Init Iteration Step
            self.train_step = 0
            self.max_step = self.cfg.optim.iters

            # Init Loss Functions
            self.losses = self.init_losses()

            # Init Solvers
            if self.cfg.stage == 'nerf':
                self.optimizers, self.schedulers, self.densifiers = self.init_nerf_solvers()
            elif self.cfg.stage == 'gs':
                self.optimizers, self.schedulers, self.densifiers = self.init_gaussian_solvers()
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)

            # Export generated images
            if cfg.log.check:
                check_dir = make_path(self.exp_path / 'snapshots' / 'check')
                logger.info('Export generated samples...')
                self.export_samples(check_dir)
                if hasattr(self.diffusion, 'tp_scheduler'):
                    self.diffusion.tp_scheduler.draw_curves(check_dir)
        else:
            self.train_step = 0

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    @torch.no_grad()
    def export_samples(self, save_dir, num_samples=2):
        from core.guidance.stable_diffusion import BasicScoreDistillation
        from core.guidance.controlnet import BasicControlNetScoreDistillation
        # ControlNet
        if isinstance(self.diffusion, BasicControlNetScoreDistillation):
            cond_inputs = None
            # export smpl conditions
            for cond_name in self.controlnet_condition:
                conds_multi_view = self.smpl_prompt.write_video(
                    save_dir=save_dir,
                    cond_type=cond_name,
                )
                if num_samples == 2:
                    conds_multi_view = [conds_multi_view[0], conds_multi_view[2]]
                if cond_inputs is None:
                    cond_inputs = [[item] for item in conds_multi_view]
                else:
                    for i, cond_image in enumerate(conds_multi_view):
                        cond_inputs[i].append(cond_image)
            # export control images
            if self.cfg.log.check_sd:
                for i, conds in enumerate(cond_inputs):
                    self.diffusion.pipe(
                        prompt=self.cfg.guide.text,
                        negative_prompt=self.cfg.guide.negative_text,
                        image=conds,
                        height=self.diffusion_image_height,
                        width=self.diffusion_image_width,
                    ).images[0].save(osp.join(save_dir, f'control_{i}.jpg'))
                    for cond_name, cond in zip(self.controlnet_condition, conds):
                        cond.save(osp.join(save_dir, f'{cond_name}_{i}.jpg'))
        # SD
        elif isinstance(self.diffusion, BasicScoreDistillation) and \
          self.cfg.log.check_sd:
            cfg = self.cfg.guide.guidance_scale

            texts = [self.cfg.guide.text]
            negative_texts = [self.cfg.guide.negative_text]

            if self.cfg.prompt.text_augmentation:
                texts += self.view_prompt.texts
                negative_texts = negative_texts * len(self.view_prompt.texts)

            for i, (prompt, negative_prompt) in enumerate(zip(texts, negative_texts)):
                view_name = '' if i == 0 else f'_view{i}'
                # cfg = 7.5
                self.diffusion.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=7.5,
                ).images[0].save(osp.join(save_dir, f'sd_7.5{view_name}.jpg'))
                # cfg = default
                self.diffusion.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=cfg,
                ).images[0].save(osp.join(save_dir, f'sd_{cfg}{view_name}.jpg'))

    def init_diffusion(self):
        if not self.cfg.guide.use_sdxl:
            from core.guidance.vae import AutoEncoderSD
            from core.guidance.stable_diffusion import ScoreDistillation
            from core.guidance.controlnet import ControlNetScoreDistillation
        else:
            from core.guidance.vae import AutoEncoderSDXL as AutoEncoderSD
            from core.guidance.stable_diffusion import ScoreDistillationXL as ScoreDistillation
            from core.guidance.controlnet import ControlNetScoreDistillationXL as ControlNetScoreDistillation

        # w/o SDS
        if self.cfg.log.eval_only or self.cfg.log.pretrain_only or self.cfg.log.nerf2gs:
            if self.cfg.stage == 'nerf' and self.cfg.nerf.nerf_type == 'latent':
                return AutoEncoderSD(
                    cfg=self.cfg.guide,
                    device=self.device,
                )
            else:
                return None
        # with SDS
        elif not self.use_controlnet:
            diffusion_model = ScoreDistillation(
                cfg=self.cfg.guide,
                device=self.device,
            )
        else:
            diffusion_model = ControlNetScoreDistillation(
                cfg=self.cfg.guide,
                device=self.device,
            )

        return diffusion_model

    def init_human_template(self):
        smpl_model = SMPLModel(
            device=self.device,
            model_type=self.cfg.prompt.smpl_type,
            gender=self.cfg.prompt.smpl_gender,
            age=self.cfg.prompt.smpl_age,
            use_smplx_2020_neutral=self.cfg.prompt.use_smplx_2020_neutral,
            flat_hand_mean=self.cfg.prompt.flat_hand_mean,
            batch_size=1,
        )
        smpl_model.requires_grad_(False)
        smpl_prompt = SMPLPrompt(
            cfg=self.cfg.prompt,
            smpl_model=smpl_model,
            cond_type=self.controlnet_condition,
            height=self.diffusion_image_height,
            width=self.diffusion_image_width,
        )
        return smpl_model, smpl_prompt

    def init_nerf_model(self):
        from core.nerf.nerf_model import build_NeRFNetwork

        model = build_NeRFNetwork(self.cfg.nerf)

        logger.info(f'Loaded {self.cfg.nerf.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        # logger.info(model)

        model = model.to(self.device)

        # if model.cuda_ray and self.cfg.log.pretrain_only:
        #     with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
        #         model.update_extra_state(random_sigmas=True)

        return model

    def init_nerf_solvers(self):

        solvers = {
            'optimizers': {},
            'schedulers': {},
            'densifiers': {},
        }

        optimizer, scheduler = self.model.get_optimizer(cfg=self.cfg.nerf)
        solvers['optimizers']['nerf'] = optimizer
        solvers['schedulers']['nerf'] = scheduler

        return solvers['optimizers'], solvers['schedulers'], None

    def init_gaussian_model(self):
        from core.system.scene import build_scene
        from core.system.avatar import build_gaussian_avatar

        if self.cfg.character is not None:
            
            outfit_mapping = self.character_description['outfit']
            for v in outfit_mapping.values():
                v['vertex_indices'], v['face_indices'] = self.smpl_model.get_semantic_indices(v['parts'])

            body_mapping = self.character_description['body']
            for v in body_mapping.values():
                v['vertex_indices'], v['face_indices'] = self.smpl_model.get_semantic_indices(v['parts'])

        if self.cfg.render.from_nerf is not None:
            from core.nerf.nerf_model import build_NeRFNetwork
            from core.nerf.to_point_cloud import export_point_cloud, remove_points_inside_bboxes
            nerf = build_NeRFNetwork(self.cfg.nerf).to(self.device)
            nerf_path = self.get_latest_checkpoint(self.cfg.render.from_nerf)
            state_dict = torch.load(nerf_path, map_location=self.device)
            nerf.load_state_dict(state_dict['model'])
            point_cloud = export_point_cloud(nerf, resolution=self.cfg.render.nerf_resolution, split_size=256)
            if self.cfg.render.nerf_exclusion_bboxes is not None:
                n_points = len(point_cloud.points)
                point_cloud = remove_points_inside_bboxes(point_cloud, bboxes=eval(self.cfg.render.nerf_exclusion_bboxes))
                logger.info(f'Remove {n_points - len(point_cloud.points)} points inside bboxes: {self.cfg.render.nerf_exclusion_bboxes}')
            logger.info(f'Load {self.cfg.nerf.backbone} NeRF and convert to 3d point cloud.')
        else:
            nerf = None
            point_cloud = None
        
        avatar = build_gaussian_avatar(
            cfg=self.cfg,
            smpl_prompt=self.smpl_prompt,
            point_cloud=point_cloud,
            nerf=nerf,
        )

        model = build_scene(cfg=self.cfg, avatar=avatar)
        
        if self.cfg.log.nerf2gs:
            self.nerf_guidance = build_NeRFNetwork(self.cfg.nerf).to(self.device)
            self.nerf_guidance.load_state_dict(state_dict['model'])
            self.nerf_guidance.eval()

        logger.info(f'Initialize Scene Model with #Parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(avatar)

        return model

    def init_gaussian_solvers(self):

        solvers = {
            'optimizers': {},
            'schedulers': {},
            'densifiers': None,
        }

        avatar_optimizer = self.model.avatar.get_optimizer(cfg=self.cfg)
        avatar_scheduler = None
        if isinstance(avatar_optimizer, dict):
            solvers['optimizers'].update(avatar_optimizer)
        else:
            solvers['optimizers']['avatar'] = avatar_optimizer
        solvers['schedulers']['avatar'] = avatar_scheduler

        if self.model.background is not None:
            background_optimizer = self.model.background.get_optimizer(cfg=self.cfg)
            background_scheduler = None
            solvers['optimizers']['background'] = background_optimizer
            solvers['schedulers']['background'] = background_scheduler

        if self.cfg.render.use_densifier:
            solvers['densifiers'] = {}
            avatar_densifier = self.model.avatar.get_densifier(cfg=self.cfg, optimizer=solvers['optimizers']['avatar'])
            solvers['densifiers']['avatar'] = avatar_densifier

        return solvers['optimizers'], solvers['schedulers'], solvers['densifiers']

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        # Build Camera DataSet
        dataset_kwargs = {
            'cfg': self.cfg.data,
            'device': self.device,
        }
        if self.smpl_prompt is not None:
            _CameraDataset = partial(CameraDatasetWithSMPL, **dataset_kwargs, smpl_prompt=self.smpl_prompt)
        else:
            _CameraDataset = partial(CameraDataset, **dataset_kwargs)
        # Build Training Set
        train_h, train_w = eval(str(self.cfg.data.train_h)), eval(str(self.cfg.data.train_w))
        train_h = (train_h,) if isinstance(train_h, Number) else train_h
        train_w = (train_w,) if isinstance(train_w, Number) else train_w
        train_dataloaders = [
            _CameraDataset(mode='train', H=H, W=W, eval_size=self.cfg.optim.iters).dataloader(batch_size=self.cfg.optim.batch_size)
            for H, W in zip(train_h, train_w)
        ]
        # Build Val and Test Sets
        eval_h, eval_w = self.cfg.data.eval_h, self.cfg.data.eval_w
        val_loader = _CameraDataset(mode='val', H=eval_h, W=eval_w, eval_size=self.cfg.data.eval_size).dataloader()
        test_h, test_w = self.cfg.data.test_h, self.cfg.data.test_w
        test_loader = _CameraDataset(mode='test', H=test_h, W=test_w, eval_size=self.cfg.data.full_eval_size).dataloader()
        # Return
        return {'train': train_dataloaders, 'val': val_loader, 'test': test_loader}

    def init_losses(self) -> Dict[str, Callable]:
        losses = {}
        losses['mse'] = torch.nn.MSELoss(reduction='mean')
        # losses['mse'] = torch.nn.MSELoss(reduction='sum')
        
        from core.nerf.nerf_loss import SparsityLoss
        losses['sparsity'] = SparsityLoss(cfg=self.cfg.nerf)

        if self.cfg.log.nerf2gs:
            from core.gaussian.gaussian_loss import ImageReconstructionLoss
            losses['image'] = ImageReconstructionLoss()
        
        return losses

    @torch.inference_mode()
    def init_text_embeddings(self, diffusion) -> Tuple:
        text = self.cfg.guide.text
        null_text = self.cfg.guide.null_text
        neg_text = self.cfg.guide.negative_text
        # Encoding prompts
        prompt_embeds, null_prompt_embeds, prompt_embeds_kwargs, null_prompt_embeds_kwargs = \
            diffusion.get_text_embeds(
                prompt=[text],
                negative_prompt=[null_text],
            )
        _, negative_prompt_embeds, _, _ = diffusion.get_text_embeds(
            prompt=[text],
            negative_prompt=[neg_text],
            **prompt_embeds_kwargs,
        )
        # Encoding view-dependent prompts
        viewed_prompt_embeds_list = []
        if self.cfg.prompt.text_augmentation:
            for viewed_text in self.view_prompt.texts:
                viewed_prompt_embeds, _, _, _ = diffusion.get_text_embeds(
                    prompt=[viewed_text],
                    **null_prompt_embeds_kwargs,
                )
                viewed_prompt_embeds_list.append(viewed_prompt_embeds)
        # Return
        return {
            'null': null_prompt_embeds,
            'pos': prompt_embeds,
            'neg': negative_prompt_embeds,
            'viewed': viewed_prompt_embeds_list,
        }

    def render(self, data, bg_mode=None):
        if self.cfg.stage == 'nerf':
            if self.model.training:
                if self.cfg.guide.diffusion != 'normal-adapted':
                    render_outputs = self.model.render(data=data, shading='albedo', bg_mode=bg_mode)
                else:
                    render_outputs = self.model.render(data=data, shading='normal', bg_mode=bg_mode)
            else:
                render_outputs = self.model.render(data=data, shading='albedo', bg_mode=bg_mode)
                render_outputs['normal'] = self.model.render(data=data, shading='normal', bg_mode=bg_mode)['image']

        elif self.cfg.stage == 'gs':
            if self.cfg.prompt.scene != 'canonical' or self.cfg.render.always_animate:
                smpl_observed_inputs = data['smpl_inputs']
            else:
                smpl_observed_inputs = None
            use_densifier = self.model.training and self.cfg.render.use_densifier
            render_outputs = self.model.forward(
                data=data,
                smpl_observed_inputs=smpl_observed_inputs,
                use_densifier=use_densifier,
                bg_mode=bg_mode,
            )
        
        else:
            assert 0, self.cfg.stage
        
        # render_outputs['depth'] = render_outputs['depth'] / render_outputs['depth'].max()
        
        return render_outputs

    def get_spatial_scale(self, data):
        if self.cfg.render.spatial_scale is None:
            spatial_scale = data['radius'].mean().item() * data['tanfov'].mean().item()
        else:
            spatial_scale = self.cfg.render.spatial_scale
        return spatial_scale

    def calc_sigma_loss(self, data, render_outputs, sd_inputs, selected_parts, wo_wrist:bool=True):

        loss_type = self.cfg.sigma_loss_type
        noise_range = self.cfg.sigma_noise_range

        import trimesh
        from igl import point_mesh_squared_distance
        from utils.trimesh import to_trimesh
        from core.nerf.nerf_utils import trunc_exp
        from core.nerf.nerf_loss import ce_pq_loss

        _, part_fids = self.smpl_model.get_semantic_indices(select_parts=selected_parts)

        v = data['smpl_outputs'].vertices[0].detach().cpu().numpy()
        f = self.smpl_model.model.faces[part_fids, :]

        mesh = to_trimesh(v, f)
        
        if self.cfg.sigma_num_points < 0:
            num_samples = f.shape[0]
        else:
            num_samples = self.cfg.sigma_num_points

        points, fids = mesh.sample(num_samples, return_index=True)

        # interpolate vertex normals from barycentric coordinates
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[fids],
            points=points,
        )
        bary = trimesh.unitize(bary).reshape((-1, 3, 1))
        vertex_normals = mesh.vertex_normals[mesh.faces[fids]]
        point_normals = trimesh.unitize((vertex_normals * bary).sum(axis=1))

        noises = (np.random.rand(points.shape[0], 1) - 0.5) * noise_range
        noisy_points = points + noises * point_normals

        surface_thickness = self.cfg.sigma_surface_thickness
        distances_pow2, closest_fids, _ = \
            point_mesh_squared_distance(noisy_points, mesh.vertices, mesh.faces)  # np.ndarray (N,)
        
        distance_mask = distances_pow2 ** 0.5 > surface_thickness
        new_points = noisy_points[distance_mask]
        closest_fids = closest_fids[distance_mask]

        if wo_wrist:
            wrist_fids = self.smpl_model.get_semantic_indices(select_parts=['wrists'])[1]
            in_wrist = np.vectorize(set(wrist_fids).__contains__)
            wrist_mask = in_wrist(np.array(part_fids)[closest_fids])
            new_points = new_points[~wrist_mask]

        xyzs = torch.from_numpy(np.concatenate((points, new_points), axis=0)).to(sd_inputs)

        losses = {}

        if loss_type.startswith('opacity'):
            # sigmas: shape = [N,], value = [0, +inf]
            # albedos: shape = [N, 3], value = [0, 1]
            sigmas, albedos = self.model.common_forward(xyzs)
            opacities = (1.0 - trunc_exp(-self.cfg.sigma_guidance_delta * sigmas))
            opacities_gt = torch.cat((torch.ones(points.shape[0]), torch.zeros(new_points.shape[0]))).to(sd_inputs)
            if loss_type == 'opacity_ce':
                sigma_loss = ce_pq_loss(opacities, opacities_gt)
            elif loss_type == 'opacity_mse':
                sigma_loss = self.losses['mse'](opacities, opacities_gt)
        else:
            peak = self.cfg.sigma_guidance_peak
            sigmas, albedos = self.model.local_geometry_forward(xyzs)
            if loss_type == 'mse':
                weights = torch.cat((torch.ones(points.shape[0]), torch.zeros(new_points.shape[0]))).to(sd_inputs)
                sigmas_gt = peak * (weights - 0.5) * 2
                sigma_loss = self.losses['mse'](sigmas, sigmas_gt)
            elif loss_type == 'margin':
                sigma_loss_neg = F.relu((sigmas[num_samples:] + peak), inplace=True)
                sigma_loss_pos = F.relu((peak - sigmas[:num_samples]), inplace=True)
                sigma_loss = (sigma_loss_neg ** 2.0).mean() + (sigma_loss_pos ** 2.0).mean()

        losses['sigma_loss'] = sigma_loss * self.cfg.lambda_sigma_sigma

        if self.cfg.lambda_sigma_albedo > 0.0:
            albedo_loss = albedos[:num_samples].var(dim=0).sum()
            losses['albedo_loss'] = albedo_loss * self.cfg.lambda_sigma_albedo

        if self.cfg.lambda_sigma_normal > 0.0:
            normals = self.model.normal(xyzs[:num_samples])
            normals_gt = torch.from_numpy(point_normals).to(normals).detach()
            normal_loss = (1.0 - normals.dot(normals_gt).abs()).mean()
            losses['normal_loss'] = normal_loss * self.cfg.lambda_sigma_normal

        if self.time_to_snapshot:
            _sigmas = torch.log(render_outputs['sigmas'].detach())
            _min = _sigmas.min().item()
            _max = _sigmas.max().item()
            _std = _sigmas.std().item()
            logger.info(f'sigmas: min={_min:.2f}, max={_max:.2f}, std={_std:.2f}')
            _sigmas = sigmas[:num_samples].detach()
            _min = _sigmas.min().item()
            _max = _sigmas.max().item()
            _std = _sigmas.std().item()
            logger.info(f'{selected_parts}_mesh_sigmas: min={_min:.2f}, max={_max:.2f}, std={_std:.2f}')
            _sigmas = sigmas[num_samples:].detach()
            _min = _sigmas.min().item()
            _max = _sigmas.max().item()
            _std = _sigmas.std().item()
            logger.info(f'{selected_parts}_empty_sigmas: min={_min:.2f}, max={_max:.2f}, std={_std:.2f}')
    
        return losses

    def train(self):
        logger.info('Starting training ^_^')
        self.model.train()

        pbar = tqdm(total=self.max_step, initial=self.train_step, bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        data_manager = DataLoaderManager(
            loaders=self.dataloaders['train'],
            milestones=self.cfg.data.grid_milestone,
            progressive=self.cfg.data.progressive_grid,
            hook_func=lambda iterator, train_step, max_step: iterator.loader._data.set_training_ratio(train_step, max_step),
        )

        try:
            while self.train_step < self.max_step:
                # Update Density Grid
                if self.cfg.stage == 'nerf':
                    if self.model.cuda_ray and self.train_step % self.cfg.nerf.update_extra_interval == 0:
                        with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                            self.model.update_extra_state()

                # Update Iteration Step (don't modified!)
                self.train_step += 1
                if self.train_step % 50 == 0:
                    pbar.update(50)

                # Keep going over dataloader until finished the required number of iterations
                data = data_manager(self.train_step, self.max_step)

                # Update Spatial Scale
                spatial_scale = self.get_spatial_scale(data)

                # Forward
                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    loss, render_outputs, sd_outputs, text = self.train_forward(data)

                # Optimizer Clear Gradient
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad()
                    if hasattr(optimizer, 'update_learning_rate'):
                        optimizer.update_learning_rate(iteration=self.train_step, spatial_scale=spatial_scale)

                # Gradient Visualization
                viz_image_gradients = False
                if self.time_to_snapshot and viz_image_gradients:
                    def _hook(grad):
                        grad_ = grad.detach().clone()
                        render_outputs['image_gradients'] = grad_
                    render_outputs['image'].register_hook(_hook)
                
                self.scaler.scale(loss).backward()

                # Densification
                if self.densifiers is not None:
                    self.model.densify(
                        densifiers=self.densifiers,
                        render_outputs=render_outputs,
                        spatial_scale=spatial_scale,
                        train_step=self.train_step,
                    )

                # Optimizer Step
                for optimizer in self.optimizers.values():
                    self.scaler.step(optimizer)
                self.scaler.update()

                # Scheduler Step
                for scheduler in self.schedulers.values():
                    if scheduler is not None:
                        scheduler.step()

                if self.time_to_checkpoint:
                    self.save_checkpoint(full=False)

                if self.time_to_evaluate:
                    logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
                    self.evaluate(self.dataloaders['val'], self.snapshot_eval_path)
                    self.model.train()

                if self.time_to_snapshot:
                    self.snapshot(data, render_outputs, sd_outputs)
                    
                    t = sd_outputs['timestep'].item()
                    azim = data['azimuth'][0].item()
                    elev = data['elevation'][0].item()
                    logger.info(f'iter={self.train_step:06d}, t={t:04d}, text={text}, azim={azim:.1f}, elev={elev:.1f}, scale={spatial_scale:.2f}')
                    
                    loss_info = f'total_loss={loss.item():.2f}'
                    if 'regularizations' in render_outputs:
                        for k, v in render_outputs['regularizations'].items():
                            loss_info += f', {k}={v.item():.2f}'
                    logger.info(loss_info)
        
        except RuntimeError as e:
            self.save_checkpoint(full=True)
            self.full_eval()
            logger.info(e)
            return
        
        logger.info('Finished Training ^_^')
        logger.info('Evaluating and save the last model...')
        self.save_checkpoint(full=False)

        self.full_eval()

        logger.info('\tDone!')

    def train_forward(self, data: Dict[str, Any]):
        # Render Images
        render_outputs = self.render(data=data)
        sd_inputs = render_outputs['image'].permute(0, 3, 1, 2).contiguous()

        # Text Embeddings
        if self.cfg.prompt.text_augmentation:
            # assert batch_size == 1
            view_index = self.view_prompt(
                azim=data['azimuth'],
                elev=data['elevation'],
            ).item()
            self.text_embeds_dict['text'] = self.text_embeds_dict['viewed'][view_index]
            text = self.view_prompt.texts[view_index]
        else:
            self.text_embeds_dict['text'] = self.text_embeds_dict['pos']
            text = self.cfg.guide.text

        sd_kwargs = {
            'inputs': sd_inputs,
            'text_embeds_dict': self.text_embeds_dict,
            'train_step': self.train_step,
            'max_iteration': self.max_step,
            'grad_viz': self.cfg.guide.grad_viz and self.time_to_snapshot,
            'scaler': self.scaler,
        }
        if self.cfg.guide.grad_rgb_clip_mask_guidance:
            if self.cfg.stage == 'nerf':
                sd_kwargs['mask_inputs'] = render_outputs['weights_sum'].permute(0, 3, 1, 2).contiguous()
            else:
                assert 0
        if self.use_controlnet:
            sd_kwargs['cond_inputs'] = data['cond_images']
        
        # Diffusion Loss
        sd_outputs = self.diffusion(**sd_kwargs)
        diffusion_loss = sd_outputs['diffusion_loss'] * self.cfg.guide.lambda_guidance

        # Regularization Loss
        regularizations = {}

        # Regularization - Sigma Loss
        if self.cfg.stage == 'nerf':
            sigma_select_parts = None
            
            if self.cfg.use_sigma_guidance:
                if random() <= self.cfg.sigma_prob:
                    # sigma_select_parts = choice(self.cfg.predefined_body_parts.split(','))
                    sigma_select_parts = self.cfg.predefined_body_parts.split(',')
            elif (
                (self.cfg.use_sigma_hand_guidance and 'hand' in data['body_part']) or
                (self.cfg.use_sigma_face_guidance and 'face' == data['body_part']) 
            ):
                sigma_select_parts = data['body_part']
            
            if sigma_select_parts is not None:
                regularizations.update(self.calc_sigma_loss(
                    data, render_outputs, sd_inputs, selected_parts=sigma_select_parts,
                ))

        # Regularization - Sparsity Loss
        if self.losses['sparsity'].available:
            if self.cfg.stage == 'nerf':
                regularizations['sparsity_loss'] = self.losses['sparsity'](
                    pred_ws=render_outputs['weights_sum'],
                    current_step=self.train_step,
                    max_iteration=self.max_step,
                )
            elif self.cfg.stage == 'gs':
                regularizations['sparsity_loss'] = self.losses['sparsity'](
                    pred_ws=render_outputs['alpha'],
                    current_step=self.train_step,
                    max_iteration=self.max_step,
                )

        render_outputs['regularizations'] = regularizations

        # Total Loss
        total_loss = 0.0
        total_loss += diffusion_loss
        for k in regularizations.keys():
            if k.endswith('_loss'):
                total_loss += regularizations[k]

        return total_loss, render_outputs, sd_outputs, text

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader,
        save_path: Path,
        save_as_video: bool = False,
        save_as_image: bool = True,
        save_one_image: bool = False,
        save_condition: bool = False,
        use_tqdm: bool = False,
        output_types=('image', 'image_fg', 'depth', 'alpha', 'normal'),
        output_types_for_video=('image', 'depth', 'alpha', 'normal'),
        output_types_for_gif=('image_fg', 'normal'),
        image_name: str = None,
        video_name: str = None,
        video_fps: int = 25,
        **kwargs,
    ):
        self.model.eval()

        save_path.mkdir(exist_ok=True)

        video_writers = {}
        gif_writers = {}
        if save_as_video:
            from utils.video import VideoWriterPyAV, VideoWriterPIL
            for output_type in output_types_for_video:
                video_filename = output_type if video_name is None else output_type + '_' + video_name
                video_writers[output_type] = VideoWriterPyAV(
                    video_path=save_path / f"{self.train_step:06d}_{video_filename}.mp4",
                    fps=video_fps,
                )
            for output_type in output_types_for_gif:
                video_filename = output_type if video_name is None else output_type + '_' + video_name
                gif_writers[output_type] = VideoWriterPIL(
                    video_path=save_path / f"{self.train_step:06d}_{video_filename}.gif",
                    fps=video_fps,
                )

        if use_tqdm:
            pbar = tqdm(total=len(dataloader), bar_format='{desc}: {percentage:3.0f} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for i, data in enumerate(dataloader):

            if save_condition:
                cond_images = data['cond_images'][0]
                (save_path / 'condition').mkdir(exist_ok=True)
                cond_images.save(save_path / 'condition' / f'{i:04d}.png')

            render_outputs = self.render(data=data, bg_mode=self.cfg.data.eval_bg_mode)

            for output_type in output_types:
                if output_type not in render_outputs.keys():
                    continue

                image = render_outputs[output_type]

                if output_type.startswith('image') and image.shape[3] == 4:
                    image = self.latent_to_image(image, color_channel=3)

                if output_type == 'depth':
                    image = image / 3.0
                elif output_type == 'normal':
                    image = torch.cat([image, render_outputs['alpha']], dim=3)
                elif output_type == 'image_fg':
                    image = self.concat_alpha(image, alpha=render_outputs['alpha'], color_channel=3)
                
                image = tensor2image(image, color_channel=3)

                if save_as_image:
                    dirname = output_type if image_name is None else output_type + '_' + image_name
                    (save_path / dirname).mkdir(exist_ok=True)
                    if output_type == 'image':
                        image.save(save_path / dirname / f"{self.train_step:06d}_{i:03d}.jpg", quality=95)
                    else:
                        image.save(save_path / dirname / f"{self.train_step:06d}_{i:03d}.png")
                
                if save_as_video:
                    if output_type in video_writers:
                        video_writers[output_type].write(image)
                    if output_type in gif_writers:
                        gif_writers[output_type].write(image)

                if save_one_image and i == 0:
                    image.save(save_path / f"{self.train_step:06d}_{output_type}.png")

            if use_tqdm:
                pbar.update(1)
        
        if save_as_video:
            for video_writer in video_writers.values():
                video_writer.release()
            for gif_writer in gif_writers.values():
                gif_writer.release()

    def full_eval(self, fast:bool=False, rgb_only:bool=True):

        if not self.cfg.log.eval_only:
            dirname = f'{self.cfg.data.test_h}x{self.cfg.data.test_w}'
        else:
            dirname = f'{self.cfg.data.test_h}x{self.cfg.data.test_w}_{self.cfg.prompt.scene}'
            if self.cfg.data.eval_body_part is not None:
                dirname += f'_{self.cfg.data.eval_body_part}'
            if self.cfg.log.eval_dirname is not None:
                dirname += f'_{self.cfg.log.eval_dirname}'

        if rgb_only:
            output_types = ('image', 'image_fg')
            ouptut_types_for_video = ('image',)
            output_types_for_gif = ()
        else:
            output_types = ('image', 'image_fg', 'depth', 'alpha', 'normal')
            ouptut_types_for_video = ('image', 'depth', 'alpha', 'normal')
            output_types_for_gif = ('image_fg', 'normal')

        final_renders_path = make_path(self.exp_path / 'results' / dirname)

        logger.info(f'Fully evaluating the last model, iteration #{self.train_step}, path: {str(final_renders_path)}')
        self.evaluate(
            dataloader=self.dataloaders['test'],
            save_path=final_renders_path,
            save_as_video=self.cfg.data.eval_save_video,
            save_as_image=self.cfg.data.eval_save_image,
            output_types=output_types,
            ouptut_types_for_video=ouptut_types_for_video,
            output_types_for_gif=output_types_for_gif,
            save_one_image=False,
            use_tqdm=True,
            video_fps=self.cfg.data.eval_video_fps,
        )
        if fast:
            return
    
    def pretrain(self):
        logger.info('Starting pretraining ^_^')
        self.model.train()

        pbar = tqdm(total=self.max_step, initial=self.train_step, bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        data_manager = DataLoaderManager(
            loaders=self.dataloaders['train'],
            milestones=self.cfg.data.grid_milestone,
            progressive=self.cfg.data.progressive_grid,
            hook_func=lambda iterator, train_step, max_step: iterator.loader._data.set_training_ratio(train_step, max_step),
        )

        while self.train_step < self.max_step:

            # Update Density Grid
            if self.cfg.stage == 'nerf':
                if self.model.cuda_ray and self.train_step % self.cfg.nerf.update_extra_interval == 0:
                    with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                        self.model.update_extra_state(random_sigmas=self.train_step == 0)

            # Update Iteration Step (don't modified!)
            self.train_step += 1
            if self.train_step % 50 == 0:
                pbar.update(50)

            # Keep going over dataloader until finished the required number of iterations
            data = data_manager(self.train_step, self.max_step)

            # Update Spatial Scale
            spatial_scale = self.get_spatial_scale(data)

            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                loss, render_outputs, visual_outputs = self.pretrain_forward(data)
                
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
                if hasattr(optimizer, 'update_learning_rate'):
                    optimizer.update_learning_rate(spatial_scale=spatial_scale, iteration=self.train_step)

            self.scaler.scale(loss).backward()

            # Densification
            if self.densifiers is not None:
                self.model.densify(
                    densifiers=self.densifiers,
                    render_outputs=render_outputs,
                    spatial_scale=spatial_scale,
                    train_step=self.train_step,
                )

            for optimizer in self.optimizers.values():
                self.scaler.step(optimizer)
            self.scaler.update()

            for scheduler in self.schedulers.values():
                if scheduler is not None:
                    scheduler.step()

            if self.time_to_checkpoint:
                self.save_checkpoint(full=False)

            if self.time_to_evaluate:
                logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
                self.evaluate(self.dataloaders['val'], self.snapshot_eval_path)
                self.model.train()

            if self.time_to_snapshot:
                self.snapshot_image(render_outputs['image'], filename='image', color_channel=3)
                self.snapshot_image(visual_outputs['mask'], filename='mask', color_channel=1)

                if 'cond_images' in data:
                    self.snapshot_condition(visual_outputs['depth'], filename=self.controlnet_condition[0])
                
                logger.info(f'train_step={self.train_step:06d}, spatial_scale={spatial_scale:.2f}')
                
                loss_info = f'total_loss={loss.item():.2f}'
                if 'regularizations' in render_outputs:
                    for k, v in render_outputs['regularizations'].items():
                        loss_info += f', {k}={v.item():.2f}'
                logger.info(loss_info)

        logger.info('Finished Training ^_^')
        logger.info('Evaluating and save the last model...')
        self.save_checkpoint(full=False)

        self.full_eval()

        logger.info('\tDone!')

    def pretrain_forward(self, data):

        render_outputs = self.render(data=data)
        render_image = render_outputs['image'].permute(0, 3, 1, 2).contiguous()
        render_depth = render_outputs['depth'].permute(0, 3, 1, 2).contiguous()
        render_ws = render_outputs['weights_sum'].permute(0, 3, 1, 2).contiguous()
        device = render_image.device

        smpl_depth = np.nan_to_num(data['cond_images'][0], posinf=0.0, neginf=0.0)  # np.ndarray, [H, W]

        visual_outputs = {}
        visual_outputs['depth'] = Image.fromarray(
            (255 * smpl_depth / np.max(smpl_depth)).clip(0, 255).astype(np.uint8),
            mode="L",
        )

        smpl_depth = torch.from_numpy(smpl_depth).to(device, dtype=torch.float)  # tensor, [H, W]
        smpl_depth = smpl_depth.unsqueeze(0).unsqueeze(0)  # tensor, [1, 1, H, W]

        if render_depth.shape[-2:] != smpl_depth.shape[-2:]:
            smpl_depth = torch.nn.functional.interpolate(smpl_depth, size=render_depth.shape[-2:], mode='bicubic')

        smpl_mask = (smpl_depth > 1e-6).float()
        visual_outputs['mask'] = smpl_mask

        loss_mask = self.losses['mse'](render_ws, smpl_mask)

        loss_depth = self.losses['mse'](render_depth, smpl_depth)

        # if render_image.shape[1] == 3:
        #     loss_image = self.losses['mse'](render_image, smpl_mask * 0.5)
        # else:
        #     loss_image = self.losses['mse'](render_image, smpl_mask.expand_as(render_image))
        loss_image = 0.0
        
        loss = loss_mask + loss_depth + loss_image

        return loss, render_outputs, visual_outputs

    def pretrain_nerf2gs(self):
        logger.info('Starting nerf-to-gs pretraining ^_^')
        self.model.train()

        pbar = tqdm(total=self.max_step, initial=self.train_step, bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        data_manager = DataLoaderManager(
            loaders=self.dataloaders['train'],
            milestones=self.cfg.data.grid_milestone,
            progressive=self.cfg.data.progressive_grid,
            hook_func=lambda iterator, train_step, max_step: iterator.loader._data.set_training_ratio(train_step, max_step),
        )

        while self.train_step < self.max_step:

            # Update Iteration Step (don't modified!)
            self.train_step += 1
            if self.train_step % 50 == 0:
                pbar.update(50)

            # Keep going over dataloader until finished the required number of iterations
            data = data_manager(self.train_step, self.max_step)

            # Update Spatial Scale
            spatial_scale = self.get_spatial_scale(data)

            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                loss, render_outputs, nerf_outputs = self.pretrain_nerf2gs_forward(data)
                
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
                if hasattr(optimizer, 'update_learning_rate'):
                    optimizer.update_learning_rate(spatial_scale=spatial_scale, iteration=self.train_step)

            self.scaler.scale(loss).backward()

            # Densification
            if self.densifiers is not None:
                self.model.densify(
                    densifiers=self.densifiers,
                    render_outputs=render_outputs,
                    spatial_scale=spatial_scale,
                    train_step=self.train_step,
                )

            for optimizer in self.optimizers.values():
                self.scaler.step(optimizer)
            self.scaler.update()

            for scheduler in self.schedulers.values():
                if scheduler is not None:
                    scheduler.step()

            if self.time_to_checkpoint:
                self.save_checkpoint(full=False)

            if self.time_to_evaluate:
                logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
                self.evaluate(self.dataloaders['val'], self.snapshot_eval_path)
                self.model.train()

            if self.time_to_snapshot:
                self.snapshot_image(
                    torch.cat([render_outputs['image_fg'].detach(), render_outputs['alpha'].detach()], dim=3),
                    filename='image',
                    color_channel=3,
                )
                self.snapshot_image(
                    torch.cat([nerf_outputs['image_fg'], nerf_outputs['alpha']], dim=3),
                    filename='target',
                    color_channel=3,
                )

                logger.info(f'train_step={self.train_step:06d}, spatial_scale={spatial_scale:.2f}')
                
                loss_info = f'total_loss={loss.item():.2f}'
                if 'regularizations' in render_outputs:
                    for k, v in render_outputs['regularizations'].items():
                        loss_info += f', {k}={v.item():.2f}'
                logger.info(loss_info)

        logger.info('Finished Training ^_^')
        logger.info('Evaluating and save the last model...')
        self.save_checkpoint(full=False)

        self.full_eval()

        logger.info('\tDone!')

    def pretrain_nerf2gs_forward(self, data):

        render_outputs = self.render(data=data)
        render_image = render_outputs['image_fg'].permute(0, 3, 1, 2).contiguous()
        # render_depth = render_outputs['depth'].permute(0, 3, 1, 2).contiguous()
        # device = render_image.device

        with torch.no_grad():
            nerf_outputs = self.nerf_guidance.render(data=data, shading='albedo')
            # nerf_outputs['normal'] = self.nerf_guidance.render(data=data, shading='normal')['image']
            nerf_image = nerf_outputs['image_fg'].permute(0, 3, 1, 2).contiguous()

        loss_image = self.losses['image'](render_image, nerf_image)
        
        loss = loss_image

        return loss, render_outputs, nerf_outputs
