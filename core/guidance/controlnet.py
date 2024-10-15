import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple
from PIL import Image
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel

from configs import GuideConfig
from .stable_diffusion import ScoreDistillation, ScoreDistillationXL


class BasicControlNetScoreDistillation:
    def __init__(self, cfg) -> None:
        self.controlnet = self.pipe.controlnet
        self.cond_processors = []  # TODO
        conditioning_scale = cfg.controlnet_scale
        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(conditioning_scale, float):
            conditioning_scale = [conditioning_scale] * len(self.controlnet.nets)
        self.conditioning_scale = conditioning_scale

    def calc_condition(self, image_path):
        results = []
        image = load_image(image_path)
        for processor in self.cond_processors:
            # process image with each condition processor
            cond = processor(image)
            # store the result
            results.append(cond)
        return results

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = [image]
        if not isinstance(image, torch.Tensor):
            if isinstance(image[0], Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=Image.Resampling.LANCZOS))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        if image.shape[0] == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        return image.to(device=device, dtype=dtype)

    def prepare_condition(self, cond_inputs, cond_width, cond_height, batch_size, dtype) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(self.controlnet, ControlNetModel):
            cond_inputs = self.prepare_image(
                cond_inputs, width=cond_width, height=cond_height, batch_size=batch_size, 
                num_images_per_prompt=1, device=self.device, dtype=dtype,
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            cond_inputs_temp = []
            for cond_input in cond_inputs:
                cond_input = self.prepare_image(
                    cond_input, width=cond_width, height=cond_height, batch_size=batch_size,
                    num_images_per_prompt=1, device=self.device, dtype=dtype,
                )
                cond_inputs_temp.append(cond_input)
            cond_inputs = cond_inputs_temp
        return cond_inputs


class ControlNetScoreDistillation(ScoreDistillation, BasicControlNetScoreDistillation):
    from diffusers import StableDiffusionControlNetPipeline
    pipe: StableDiffusionControlNetPipeline

    def __init__(self, cfg: GuideConfig, device: torch.device):
        super().__init__(cfg, device=device)
        BasicControlNetScoreDistillation.__init__(self, cfg)

    def _predict(self, latents_model_input, text_embeddings, cond_inputs):

        # Default height and width to unet
        _, _, latent_height, latent_width = latents_model_input.shape
        cond_height = latent_height * self.vae_scale_factor
        cond_width = latent_width * self.vae_scale_factor

        controlnet_cond = self.prepare_condition(
            cond_inputs,
            cond_height=cond_height,
            cond_width=cond_width,
            batch_size=latents_model_input.size(0),
            dtype=latents_model_input.dtype,
        )

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents_model_input,  # [2, 4, 64, 64]
            timestep=self.timestep,  # [1,]
            encoder_hidden_states=text_embeddings,  # [2, 77, 768]
            controlnet_cond=controlnet_cond,  # [1, 3, 512, 512]
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )

        return self.unet(
            latents_model_input,
            timestep=self.timestep,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]


class ControlNetScoreDistillationXL(ScoreDistillationXL, BasicControlNetScoreDistillation):
    from diffusers import StableDiffusionXLControlNetPipeline
    pipe: StableDiffusionXLControlNetPipeline

    def __init__(self, cfg: GuideConfig, device: torch.device):
        super().__init__(cfg, device=device)
        BasicControlNetScoreDistillation.__init__(self, cfg)

    def _predict(self, latents_model_input, text_embeddings, cond_inputs, guess_mode: bool = False):

        # Default height and width to unet
        batch_size, _, latent_height, latent_width = latents_model_input.shape
        cond_height = latent_height * self.vae_scale_factor
        cond_width = latent_width * self.vae_scale_factor

        prompt_embeds, added_cond_kwargs = self.prepare_sdxl_inputs(
            text_embeddings=text_embeddings,
            batch_size=batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
        )

        global_pool_conditions = (
            self.controlnet.config.global_pool_conditions
            if isinstance(self.controlnet, ControlNetModel)
            else self.controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # controlnet(s) inference
        if guess_mode and self.do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latents_model_input.chunk(2)[1]
            control_model_input = self.scheduler.scale_model_input(control_model_input, self.timestep)
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            controlnet_added_cond_kwargs = {
                "text_embeds": added_cond_kwargs["text_embeds"].chunk(2)[1],
                "time_ids": added_cond_kwargs["time_ids"].chunk(2)[1],
            }
        else:
            control_model_input = latents_model_input
            controlnet_prompt_embeds = prompt_embeds
            controlnet_added_cond_kwargs = added_cond_kwargs

        # prepare condition images
        cond_inputs = self.prepare_condition(
            cond_inputs=cond_inputs,
            cond_height=cond_height,
            cond_width=cond_width,
            batch_size=latents_model_input.size(0),
            dtype=latents_model_input.dtype,
        )

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            self.timestep,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=cond_inputs,
            conditioning_scale=self.conditioning_scale,
            guess_mode=guess_mode,
            added_cond_kwargs=controlnet_added_cond_kwargs,
            return_dict=False,
        )

        if guess_mode and self.do_classifier_free_guidance:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        # predict the noise residual
        return self.unet(
            latents_model_input,
            timestep=self.timestep,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
