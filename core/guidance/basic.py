import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Union, List
from diffusers import DDIMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from random import random
import numpy as np
from huggingface_hub import hf_hub_download
from loguru import logger
# from transformers import logging
# logging.set_verbosity_error()
from configs import GuideConfig

from .pgc import build_grad_hook_func, build_pgc_hook_func
from .ism.guidance.sd_step import ddim_step
from .time_prior import TimePrioritizedScheduler

DTYPE_CARDS = {
    'fp16': torch.float16,
    'fp32': torch.float32,
}

MODEL_CARDS = {
    # Stable-Diffusion
    'sd14': "CompVis/stable-diffusion-v1-4",
    # 'sd15': "runwayml/stable-diffusion-v1-5",
    'sd15': "stable-diffusion-v1-5/stable-diffusion-v1-5",
    'sd20b': "stabilityai/stable-diffusion-2-base",
    'sd20': "stabilityai/stable-diffusion-2",
    'sd21b': "stabilityai/stable-diffusion-2-1-base",
    'sd21': "stabilityai/stable-diffusion-2-1",
    # HumanNorm
    "normal-adapted": "xanderhuang/normal-adapted-sd1.5",
    "depth-adapted": "xanderhuang/depth-adapted-sd1.5",
    # Stable-Diffusion-XL
    'sdxl10': "stabilityai/stable-diffusion-xl-base-1.0",
}

MODEL_FP16_KWARGS = {
    'sd15': {'revision': 'fp16'},
    'sd21': {'revision': 'fp16'},
    'sdxl10': {'variant': 'fp16'},
}

CONTROLNET_CARDS = {
    'sd15_v10': {
        # ControlNet-v1.0
        'pose': "lllyasviel/sd-controlnet-openpose",
        'depth': "lllyasviel/sd-controlnet-depth",
        'canny': "lllyasviel/sd-controlnet-canny",
        'seg': "lllyasviel/sd-controlnet-seg",
        'normal': "lllyasviel/sd-controlnet-normal",
        # 'pose': "fusing/stable-diffusion-v1-5-controlnet-openpose",
        # 'depth': "fusing/stable-diffusion-v1-5-controlnet-depth",
        # 'canny': "fusing/stable-diffusion-v1-5-controlnet-canny",
        # 'normal': "fusing/stable-diffusion-v1-5-controlnet-normal",
        # 'seg': "fusing/stable-diffusion-v1-5-controlnet-seg",
    },
    'sd15': {
        # ControlNet-v1.1
        'pose': "lllyasviel/control_v11p_sd15_openpose",
        'depth': "lllyasviel/control_v11f1p_sd15_depth",
        'canny': "lllyasviel/control_v11p_sd15_canny",
        'normal': "lllyasviel/control_v11p_sd15_normalbae",
    },
    'sd21': {
        'pose': "thibaud/controlnet-sd21-openposev2-diffusers",
        'depth': "thibaud/controlnet-sd21-depth-diffusers",
        'canny': "thibaud/controlnet-sd21-canny-diffusers",
        'normal': "thibaud/controlnet-sd21-normal-diffusers",
    },
    'sd21b': {
        'pose': "thibaud/controlnet-sd21-openposev2-diffusers",
        'depth': "thibaud/controlnet-sd21-depth-diffusers",
        'canny': "thibaud/controlnet-sd21-canny-diffusers",
        'normal': "thibaud/controlnet-sd21-normal-diffusers",
    },
    'sdxl10': {
        'pose': "thibaud/controlnet-openpose-sdxl-1.0",
        'depth': "diffusers/controlnet-depth-sdxl-1.0",
        'canny': "diffusers/controlnet-canny-sdxl-1.0",
    }
}

CIVITAI_LORA_ROOT = "/comp_robot/huangyukun/model/lora/"
CIVITAI_CKPT_ROOT = "/comp_robot/huangyukun/model/stable-diffusion"


def get_model_name_or_path(model_name, use_controlnet: bool, use_sdxl: bool):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, \
        StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
    model_name_or_path = model_name
    from_single_file = False
    if model_name in MODEL_CARDS:
        model_name_or_path = MODEL_CARDS[model_name]
        if use_sdxl:
            pipeline_cls = StableDiffusionXLControlNetPipeline if use_controlnet else StableDiffusionXLPipeline
        else:
            pipeline_cls = StableDiffusionControlNetPipeline if use_controlnet else StableDiffusionPipeline
    else:
        logger.warning(f'Please carefully check pipeline type for {model_name}')
        pipeline_cls = StableDiffusionControlNetPipeline if use_controlnet else StableDiffusionPipeline
    if model_name_or_path.endswith('.safetensors') or model_name_or_path.endswith('.ckpt'):
        if not osp.isfile(model_name):
            model_name_or_path = osp.join(CIVITAI_CKPT_ROOT, model_name)
        from_single_file = True
    return model_name_or_path, pipeline_cls, from_single_file


def build_stable_diffusion_pipeline(
    model_name: str,
    torch_dtype: torch.dtype,
    fp16: bool,
    controlnet=None,
    use_sdxl=False,
    **kwargs,
):
    # Initial kwargs
    extra_kwargs = {
        'torch_dtype': torch_dtype,
        'local_files_only': False,
        "feature_extractor": None,
        "requires_safety_checker": False,
    }
    use_controlnet = controlnet is not None
    model_name_or_path, pipeline_cls, from_single_file = \
        get_model_name_or_path(model_name, use_controlnet=use_controlnet, use_sdxl=use_sdxl)
    class_name = pipeline_cls.__name__
    logger.info(f'Loading {class_name} from {model_name_or_path}...')
    # Prepare Configs
    if use_controlnet:
        extra_kwargs['controlnet'] = controlnet
    if fp16:
        if model_name in MODEL_FP16_KWARGS:
            extra_kwargs.update(MODEL_FP16_KWARGS[model_name])
        else:
            logger.warning(f'\t Please carefully check fp16 kwargs for {model_name}')
    extra_kwargs.update(kwargs)
    # Build Pipeline
    if not from_single_file:
        pipe = pipeline_cls.from_pretrained(model_name_or_path, safety_checker=None, **extra_kwargs)
    else:
        pipe = pipeline_cls.from_single_file(model_name_or_path, safety_checker=None, **extra_kwargs)
    logger.info(f'\t Successfully loaded {class_name}!')
    if from_single_file:
        logger.warning(f'Please carefully check foundation diffusion model type for {model_name_or_path}')
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
    return pipe, model_name_or_path


def build_controlnet(
    controlnet_name: str,
    condition_types: Union[str, List[str]],
    torch_dtype: torch.dtype,
    fp16: bool = False,
):
    from diffusers.models import ControlNetModel
    from diffusers.pipelines.controlnet import MultiControlNetModel
    # Initial kwargs
    extra_kwargs = {
        'local_files_only': False,
        'torch_dtype': torch_dtype,
    }
    if fp16:
        extra_kwargs['variant'] = 'fp16'
    # Func
    def _build_controlnet(condition_type):
        model_name_or_path = CONTROLNET_CARDS[controlnet_name][condition_type]
        logger.info(f'Loding ControlNet from {model_name_or_path}...')
        controlnet = ControlNetModel.from_pretrained(model_name_or_path, **extra_kwargs)
        logger.info(f'\t Successfully loaded ControlNet!')
        return controlnet
    # Build ControlNet
    if isinstance(condition_types, str):
        condition_types = [condition_types]
    controlnets = []
    for each_type in condition_types:
        controlnets.append(_build_controlnet(each_type))
    if len(controlnets) > 1:
        controlnet = MultiControlNetModel(controlnets=controlnets)
        logger.info(f'\t Successfully loaded MultiControlNet conditioned on {condition_types}')
    else:
        controlnet = controlnets[0]
    return controlnet


def build_vae(
    model_name: str, 
    torch_dtype: torch.dtype,
    use_sdxl: bool,
):
    from diffusers import AutoencoderKL
    model_name_or_path, _, from_single_file = get_model_name_or_path(model_name, use_controlnet=False, use_sdxl=use_sdxl)
    if from_single_file:
        logger.warning(f'Please carefully check foundation diffusion model type for {model_name_or_path}')
        model_name_or_path = "runwayml/stable-diffusion-v1-5"
    logger.info(f'Loading VAE from {model_name_or_path}!')
    if use_sdxl and torch_dtype is torch.float16:
        logger.info('\t For SDXL-fp16, use fp16-fixed VAE from madebyollin/sdxl-vae-fp16-fix!')
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
    else:
        vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder="vae", torch_dtype=torch_dtype)
    logger.info(f'\t Successfully loaded VAE!')
    return vae


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class BasicStableDiffusion:
    def __init__(self, cfg: GuideConfig):
        # Init
        self.cfg = cfg
        self.torch_dtype = DTYPE_CARDS[cfg.dtype]
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
        self.rescale_noise_cfg = rescale_noise_cfg
        # Build ControlNet
        if cfg.use_controlnet:
            controlnet = build_controlnet(
                controlnet_name=cfg.controlnet,
                condition_types=cfg.controlnet_condition,
                torch_dtype=self.torch_dtype,
                fp16=cfg.controlnet_fp16,
            )
        else:
            controlnet = None
        # Build VAE
        vae = build_vae(
            model_name=cfg.diffusion,
            torch_dtype=self.torch_dtype,
            use_sdxl=cfg.use_sdxl,
        )
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # Build Pipeline
        self.pipe, self.model_name_or_path = build_stable_diffusion_pipeline(
            model_name=cfg.diffusion,
            torch_dtype=self.torch_dtype,
            fp16=cfg.diffusion_fp16,
            controlnet=controlnet,
            use_sdxl=cfg.use_sdxl,
            vae=vae,
        )
        # Load Extra Modules
        if cfg.lora_name is not None:
            self.load_lora()
        if cfg.concept_name is not None:
            self.load_concept()

    def read_token(self):
        try:
            with open('./TOKEN', 'r') as f:
                token = f.read().replace('\n', '')  # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            logger.warning('try to load hugging face access token from the default place,'
                           ' make sure you have run `huggingface-cli login`.')
            token = True
        return token

    def load_lora(self):
        lora_name = self.cfg.lora_name
        lora_model_path = osp.join(CIVITAI_LORA_ROOT, lora_name) if not osp.isfile(lora_name) else lora_name
        self.pipe.load_lora_weights(lora_model_path)
        logger.info(f'Successfully loaded LoRA from {lora_model_path}!')

    def load_concept(self):
        concept_name = self.cfg.concept_name
        repo_id_embeds = f"sd-concepts-library/{concept_name}"
        learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
        with open(token_path, 'r') as file:
            placeholder_token_string = file.read()

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}."
                f" Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds


class BasicScoreDistillation(BasicStableDiffusion):
    tp_scheduler: TimePrioritizedScheduler

    def __init__(self, cfg: GuideConfig):
        super().__init__(cfg)
        # Init
        self.loss_type = cfg.sds_loss_type
        self.weight_type = cfg.sds_weight_type
        self.initial_guidance_scale = cfg.guidance_scale
        self.guidance_adjust = cfg.guidance_adjust
        self.do_classifier_free_guidance = self.initial_guidance_scale > 1.0
        self.use_negative_text = cfg.use_negative_text
        # Default Input Size
        self.input_interpolate = cfg.input_interpolate
        self.default_latent_size = self.pipe.unet.config.sample_size
        self.default_image_size = self.default_latent_size * self.vae_scale_factor
        # Placeholder
        self.add_noise, self.get_timestep = None, None
        self.alphas, self.alphas_cumprod, self.betas = None, None, None
        self.timesteps, self.timestep, self.guidance_scale = None, None, None
        # Freeze pipeline
        self.freeze_pipeline()

    @property
    def is_denoising_mode(self):
        return self.loss_type in ('z0', 'z0_final', 'x0', 'x0_final')

    def freeze_pipeline(self):
        for _, m in self.pipe.components.items():
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad_(False)
                m.eval()

    def prepare_latents(self, inputs: Tensor):
        """
          - inputs: Tensor, [N, 3 or 4, H, W]
        """
        if inputs.size(1) == 3:
            default_size = (self.default_image_size, self.default_image_size)
            if self.input_interpolate and (inputs.shape[-2:] != torch.Size(default_size)) and not \
              (self.cfg.use_sdxl and inputs.shape[-2:] == torch.Size((768, 768))):
                inputs = F.interpolate(inputs, default_size, mode='bilinear', align_corners=False)
            if self.cfg.use_sdxl:
                assert inputs.shape[2] in (768, 1024) and inputs.shape[3] in (768, 1024), inputs.shape
            else:
                assert inputs.shape[2] in (768, 512,) and inputs.shape[3] in (768, 512,), inputs.shape
            # Encode image into latents with vae, requires grad!
            if self.loss_type in ('x0', 'x0_final'):
                with torch.no_grad():
                    latents = self.encode_images(inputs)
            else:
                latents = self.encode_images(inputs)
        else:
            default_size = (self.default_latent_size, self.default_latent_size)
            if self.input_interpolate and (inputs.shape[-2:] != torch.Size(default_size)):
                inputs = F.interpolate(inputs, default_size, mode='bilinear', align_corners=False)
            if self.cfg.use_sdxl:
                assert inputs.shape[2] in (128, 96) and inputs.shape[3] in (128, 96), inputs.shape
            else:
                assert inputs.shape[2] in (64, 96) and inputs.shape[3] in (64, 96), inputs.shape
            latents = inputs
    
        return latents, inputs

    def setup_scheduler(self, scheduler, device, num_inference_timesteps=50):
        # DreamTime
        num_train_timesteps = scheduler.config.num_train_timesteps
        tp_scheduler = TimePrioritizedScheduler(self.cfg, device=device, num_train_timesteps=num_train_timesteps,
                                                pretrained_model_name_or_path=self.model_name_or_path,
                                                scheduler='ddim' if self.loss_type == 'ism' else 'ddpm')
        # Setup Scheduler
        scheduler.set_timesteps(num_inference_timesteps)
        tp_scheduler.scheduler.set_timesteps(num_train_timesteps)
        # self.add_noise = scheduler.add_noise
        self.add_noise = tp_scheduler.add_noise
        self.get_timestep = tp_scheduler.get_timestep
        self.alphas = tp_scheduler.alphas
        self.betas = tp_scheduler.betas
        self.alphas_cumprod = tp_scheduler.alphas_cumprod
        self.timesteps = torch.flip(tp_scheduler.scheduler.timesteps, dims=(0,)).to(self.device)

        return tp_scheduler

    def get_guidance_scale(self, train_step, max_iteration):
        initial_guidance_scale = self.initial_guidance_scale
        if self.guidance_adjust == 'constant':
            guidance_scale = initial_guidance_scale
        elif self.guidance_adjust == 'uniform':
            guidance_scale = np.random.uniform(7.5, initial_guidance_scale)
        elif self.guidance_adjust == 'linear':
            guidance_delta = (initial_guidance_scale - 7.5) / (max_iteration - 1)
            guidance_scale = initial_guidance_scale - (train_step - 1) * guidance_delta
        elif self.guidance_adjust == 'linear_reverse':
            guidance_delta = (initial_guidance_scale - 7.5) / (max_iteration - 1)
            guidance_scale = 7.5 + (train_step - 1) * guidance_delta
        else:
            raise NotImplementedError
        return guidance_scale

    def preprocess(self, inputs, train_step, max_iteration, **kwargs):
        batch_size = inputs.size(0)

        # Prepare latents to be fed into unet
        latents, inputs = self.prepare_latents(inputs)

        # Adaptive guidance scale
        if 'guidance_scale' in kwargs:
            self.guidance_scale = kwargs.pop('guidance_scale')
        else:
            self.guidance_scale = self.get_guidance_scale(train_step, max_iteration)

        # Adaptive timestep
        if 'timestep' in kwargs:
            self.timestep = kwargs.pop('timestep')
        else:
            self.timestep = self.get_timestep(batch_size, train_step, max_iteration)

        return latents, inputs, kwargs

    def _predict(self, latents_model_input: Tensor, text_embeddings: Tensor, **kwargs):
        raise NotImplementedError

    def get_noise_pred(
        self,
        latents_noisy: Tensor,
        text_embeddings: Tensor,
        guidance_rescale: float = 0.0,
        **kwargs,
    ):
        # Expand the latents if we are doing classifier free guidance
        latents_model_input = torch.cat([latents_noisy] * 2) if self.do_classifier_free_guidance else latents_noisy
        # latents_model_input = self.tp_scheduler.scheduler.scale_model_input(latents_model_input, self.timestep)
        latents_model_input = self.scheduler.scale_model_input(latents_model_input, self.timestep)

        # unet forward
        noise_pred = self._predict(latents_model_input, text_embeddings, **kwargs)

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        return noise_pred
    
    def get_denoise_pred(
        self,
        latents_noisy:Tensor,
        noise_pred:Tensor,
        text_embeddings:Tensor,
        iterative:bool,
        num_timesteps=50,
        **kwargs,
    ):
        # init
        bs = latents_noisy.shape[0]  # batch size
        scheduler = self.tp_scheduler.scheduler

        # use only 50 timesteps, and find nearest of those to t
        scheduler.set_timesteps(num_timesteps)
        scheduler.timesteps_gpu = scheduler.timesteps.to(latents_noisy.device)
        large_enough_idxs = scheduler.timesteps_gpu.expand([bs, -1]) > self.timestep[:bs].unsqueeze(-1)  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = scheduler.timesteps_gpu[idxs]

        fracs = list((t / scheduler.config.num_train_timesteps).cpu().numpy())

        # get prev latent
        latents_1step = []
        latents_1orig = []
        for b in range(bs):
            step_output = scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1]
            )
            latents_1step.append(step_output["prev_sample"])
            latents_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        latents_1orig = torch.cat(latents_1orig)

        step_outputs = {
            "noise_levels": fracs,
            "latents_noisy": latents_noisy,
            "latents_1orig": latents_1orig,
            "latents_1step": latents_1step,
        }

        if iterative:
            latents_final = []
            for b, i in enumerate(idxs):
                latents = latents_1step[b : b + 1]
                for t in scheduler.timesteps[i + 1 :]:
                    # pred noise
                    noise_pred = self.get_noise_pred(
                        latents_noisy=latents,
                        text_embeddings=text_embeddings,
                        **kwargs
                    )
                    # get prev latent
                    latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
                latents_final.append(latents)
            latents_final = torch.cat(latents_final)
            step_outputs['latents_final'] = latents_final
        
        return step_outputs

    def prepare_text_embeddings(self, text_embeds_dict: dict, text_keys: tuple):
        text_embeds_list = []
        for text_key in text_keys:
            text_embeds_list.append(text_embeds_dict[text_key])
        if torch.is_tensor(text_embeds_list[0]):
            text_embeddings = torch.concat(text_embeds_list, dim=0)
            return text_embeddings
        else:
            text_embeddings = []
            pooled_text_embeddings = []
            for text_embed, pooled_text_embed in text_embeds_list:
                text_embeddings.append(text_embed)
                pooled_text_embeddings.append(pooled_text_embed)
            text_embeddings = torch.concat(text_embeddings, dim=0)
            pooled_text_embeddings = torch.concat(pooled_text_embeddings, dim=0)
            return text_embeddings, pooled_text_embeddings

    def calc_gradients(
        self,
        latents_noisy: Tensor,
        text_embeds_dict: dict,
        noise: Tensor,
        guidance_rescale: float = 0.0,
        train_step: int = None,
        max_iteration: int = None,
        **kwargs,
    ):
        # Prepare text embeddings
        if self.loss_type in ('csd', 'nfsd'):
            text_embeddings = self.prepare_text_embeddings(
                text_embeds_dict=text_embeds_dict,
                text_keys=('null', 'text', 'neg'),
            )
            latents_model_input = torch.cat([latents_noisy] * 3, dim=0)
        
        elif self.do_classifier_free_guidance:
            text_keys = ('neg', 'text') if self.use_negative_text else ('null', 'text')
            text_embeddings = self.prepare_text_embeddings(
                text_embeds_dict=text_embeds_dict,
                text_keys=text_keys,
            )
            latents_model_input = torch.cat([latents_noisy] * 2, dim=0)
        
        else:
            text_embeddings = text_embeds_dict['text']
            latents_model_input = latents_noisy

        # Expand the latents if we are doing classifier free guidance
        # latents_model_input = self.tp_scheduler.scheduler.scale_model_input(latents_model_input, self.timestep)
        latents_model_input = self.scheduler.scale_model_input(latents_model_input, self.timestep)

        # unet forward
        noise_pred = self._predict(latents_model_input, text_embeddings, **kwargs)

        # perform classifier-free guidance
        if self.loss_type in ('csd', 'nfsd'):
            noise_pred_uncond, noise_pred_text, noise_pred_negative = noise_pred.chunk(3)
        
        elif self.loss_type in ('custom',):
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text - noise_pred_uncond
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        elif self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # calc gradients
        if self.loss_type in ('sds', 'sjc', 'ism'):
            gradients = noise_pred - noise
        
        elif self.loss_type in ('sjc-red', 'custom'):
            gradients = noise_pred

        elif self.loss_type == 'csd':
            if train_step is None or max_iteration is None:
                gradients = noise_pred_text - noise_pred_uncond
            else:
                p = train_step / max_iteration
                null_scale: float = - 0.5 * p
                neg_scale: float = -1 + 0.5 * p
                gradients = noise_pred_text + (
                    null_scale * noise_pred_uncond +
                    neg_scale * noise_pred_negative
                )
        
        elif self.loss_type == 'nfsd':
            # if t >= 200:
                # e_delta = e_null - e_neg
            # else:
                # e_delta = e_null
            delta_domain = noise_pred_uncond.clone()
            delta_domain[self.timestep >= 200] -= noise_pred_negative
            delta_conditon = noise_pred_text - noise_pred_uncond
            gradients = delta_domain + self.guidance_scale * delta_conditon

        else:
            raise NotImplementedError(f"Invalid sds loss type: {self.loss_type}")

        # Weight
        if self.weight_type is not None:
            alphas = self.alphas_cumprod[self.timestep]
            if self.weight_type == 'dreamfusion':
                w = 1 - alphas
            elif self.weight_type == 'latent-nerf':
                w = (1 - alphas) * (alphas ** 0.5)
            elif self.weight_type == 'ism':
                w = (((1 - alphas) / alphas) ** 0.5)
            elif self.weight_type == 'sjc':
                w = torch.ones_like(alphas)
            else:
                raise NotImplementedError(f"Invalid sds weight type: {self.weight_type}")
            gradients *= w.reshape(-1, 1, 1, 1)
        
        if self.cfg.grad_latent_clip:
            # gradients = gradients.clamp(-1, 1)
            # std = gradients.std().item() * self.cfg.grad_latent_clip_scale
            grad_for_std = gradients.nan_to_num(0.0, 0.0, 0.0)
            std = ((grad_for_std ** 2).sum() / grad_for_std.count_nonzero()) ** 0.5 * self.cfg.grad_latent_clip_scale
            gradients = gradients.clamp(-std, std).nan_to_num(0.0)
        
        if self.cfg.grad_latent_norm:
            gradients = gradients.nan_to_num(0.0, 0.0, 0.0)
            gradients = torch.nn.functional.normalize(gradients, p=2, dim=(1, 2, 3))
        
        if self.cfg.grad_latent_nan_to_num:
            # gradients = torch.nan_to_num(gradients, nan=0.0, posinf=1.0, neginf=-1.0)
            gradients = torch.nan_to_num(gradients)

        return gradients, noise_pred, text_embeddings

    def ism_add_noise_with_cfg(self, latents, noise, ind_t, ind_prev_t, text_embeddings=None, cfg=1.0, delta_t=1, inv_steps=1, is_noisy_latent=False, eta=0.0):

        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        scheduler = self.tp_scheduler.scheduler
        assert isinstance(scheduler, DDIMScheduler)

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t])
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input, encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = unet(cur_noisy_lat_, timestep_model_input, encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t - cur_t

            cur_noisy_lat = ddim_step(scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    def ism(self, latents, text_embeds_dict, train_step, max_iteration):
        # Step 1: sample x_s with larger steps
        xs_delta_t = 200
        xs_inv_steps = 5
        denoise_guidance_scale = 1.0

        # min_step, max_step = self.tp_scheduler.min_step, self.tp_scheduler.max_step
        min_step = 20
        max_step = 500
        warmup_step = 980 - 500

        delta_t = 80
        delta_t_start = 100

        warmup_iter = int(max_iteration * 1500 / 5000)
        warm_up_rate = 1. - min(train_step / warmup_iter, 1.)

        annealing_intervals = True
        if annealing_intervals:
            current_delta_t = int(delta_t + np.ceil(warm_up_rate * (delta_t_start - delta_t)))
        else:
            current_delta_t = delta_t

        ind_t = torch.randint(min_step, max_step + int(warmup_step*warm_up_rate), size=(1,), dtype=torch.long, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

        uncond_text_embeddings = text_embeds_dict['null']
        inverse_text_embeddings = torch.stack((uncond_text_embeddings, uncond_text_embeddings), dim=0)

        _, prev_latents_noisy, pred_scores_xs = self.ism_add_noise_with_cfg(
            latents=latents,
            noise=noise,
            ind_t=ind_prev_t,
            ind_prev_t=starting_ind,
            text_embeddings=inverse_text_embeddings,
            cfg=denoise_guidance_scale,
            delta_t=xs_delta_t,
            inv_steps=xs_inv_steps,
        )
        
        # Step 2: sample x_t
        _, latents_noisy, pred_scores_xt = self.ism_add_noise_with_cfg(
            latents=prev_latents_noisy,
            noise=noise,
            ind_t=ind_t,
            ind_prev_t=ind_prev_t,
            text_embeddings=inverse_text_embeddings,
            cfg=denoise_guidance_scale,
            delta_t=current_delta_t,
            inv_steps=1,
            is_noisy_latent=True,
        )

        pred_scores = pred_scores_xt + pred_scores_xs
        noise = pred_scores[0][1]

        return latents_noisy, noise

    def __call__(
        self,
        inputs: Tensor,
        text_embeds_dict: dict,
        train_step: int,
        max_iteration: int,
        add_noise: bool = True,
        grad_viz: bool = False,
        **kwargs,
    ):
        """
            inputs: [N, 3 or 4, H, W]
            text_embeds_dict: dict
            cond_inputs: conditional preprocessed images
        """
        
        # Pixel-wise gradient operation
        if inputs.size(1) == 3:
            if self.cfg.pgc_clip_rgb >= 0:
                _hook = build_pgc_hook_func(
                    clip_value=self.cfg.pgc_clip_rgb,
                    pgc_suppress_type=self.cfg.pgc_suppress_type,
                    scaler=kwargs['scaler'],
                )
                inputs.register_hook(_hook)
            elif self.cfg.grad_rgb_clip or self.cfg.grad_rgb_norm:
                if 'mask_inputs' in kwargs:
                    mask = kwargs.pop('mask_inputs')
                else:
                    mask = None
                _hook = build_grad_hook_func(
                    grad_clip=self.cfg.grad_rgb_clip,
                    grad_norm=self.cfg.grad_rgb_norm,
                    grad_clip_scale=self.cfg.grad_rgb_clip_scale,
                    scaler=kwargs['scaler'],
                    mask=mask,
                )
                inputs.register_hook(_hook)
        if 'scaler' in kwargs:
            kwargs.pop('scaler')

        # Preprocess
        latents, inputs, kwargs = self.preprocess(
            inputs=inputs,
            train_step=train_step,
            max_iteration=max_iteration,
            **kwargs
        )

        # Predict the noise residual with unet, no grad!
        with torch.no_grad():
            if self.loss_type == 'ism':
                latents_noisy, noise = self.ism(latents, text_embeds_dict, train_step, max_iteration)
            else:
                # https://huggingface.co/learn/diffusion-course/unit4/2
                noise = torch.randn_like(latents)
                if add_noise:
                    latents_noisy = self.add_noise(latents, noise, self.timestep)
                else:
                    latents_noisy = latents

        outputs = {
            'latents': latents,
            'timestep': self.timestep,
        }

        # Predict the noise residual with unet, no grad!
        with torch.no_grad():
            # Denoise-based Gradients
            if self.is_denoising_mode:
                text_keys = ('neg', 'text') if self.use_negative_text else ('null', 'text')
                text_embeddings = self.prepare_text_embeddings(text_embeds_dict, text_keys=text_keys)
                noise_pred = self.get_noise_pred(
                    latents_noisy=latents_noisy,
                    text_embeddings=text_embeddings,
                    **kwargs,
                )
                denoise_outputs = self.get_denoise_pred(
                    latents_noisy=latents_noisy,
                    noise_pred=noise_pred,
                    text_embeddings=text_embeddings,
                    iterative=(self.cfg.grad_viz or self.loss_type.endswith('_final')),
                    **kwargs,
                )
                outputs.update(denoise_outputs)

                if self.loss_type == 'z0':
                    sources = latents
                    targets = outputs['latents_1orig']
                
                elif self.loss_type == 'z0_final':
                    sources = latents
                    targets = outputs['latents_final']
                
                elif self.loss_type == 'x0':
                    sources = inputs
                    targets = self.decode_latents(outputs['latents_1orig'])
                    
                elif self.loss_type == 'x0_final':
                    sources = inputs
                    targets = self.decode_latents(outputs['latents_final'])
                
                gradients = (sources - targets).detach()  # for viz only

            # Score-based Gradients
            else:
                gradients, noise_pred, text_embeddings = self.calc_gradients(
                    latents_noisy=latents_noisy,
                    text_embeds_dict=text_embeds_dict,
                    noise=noise,
                    train_step=train_step,
                    max_iteration=max_iteration,
                    **kwargs,
                )
                sources = latents
                targets = (sources - gradients).detach()  # for viz only

        # Calculate loss
        if self.is_denoising_mode:
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            diffusion_loss = 0.5 * F.mse_loss(sources, targets, reduction="sum") / sources.size(0)
        else:
            diffusion_loss = SpecifyGradient.apply(sources, gradients)

        outputs['sources'] = sources
        outputs['targets'] = targets
        outputs['gradients'] = gradients
        outputs['diffusion_loss'] = diffusion_loss

        if grad_viz and 'latents_final' not in outputs:
            denoise_outputs = self.get_denoise_pred(
                latents_noisy=latents_noisy,
                noise_pred=noise_pred,
                text_embeddings=text_embeddings,
                iterative=True,
                **kwargs,
            )
            outputs.update(denoise_outputs)

        return outputs


# outputs_viz = {}
# if self.cfg.grad_viz:
#     if step_outputs is None:
#         scheduler = self.tp_scheduler.scheduler
#         step_outputs = scheduler.step(noise_pred, timestep=timestep.item(), sample=latents_noisy)
#     # outputs_viz['latents_1step'] = step_outputs['prev_sample']
#     outputs_viz['latents_1orig'] = step_outputs['pred_original_sample']
#     outputs_viz['latents_noisy'] = latents_noisy
#     outputs_viz['noise_residual'] = noise_residual
#     outputs_viz['noise_pred'] = noise_pred
#     # outputs_viz['noise'] = noise
