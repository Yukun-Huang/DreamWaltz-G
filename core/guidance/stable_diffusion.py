import torch
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple

from configs import GuideConfig
from .basic import BasicScoreDistillation
from .vae import AutoEncoderSD, AutoEncoderSDXL


class ScoreDistillation(BasicScoreDistillation, AutoEncoderSD):
    from diffusers import StableDiffusionPipeline, PNDMScheduler, UNet2DConditionModel
    pipe: StableDiffusionPipeline
    scheduler: PNDMScheduler
    unet: UNet2DConditionModel

    def __init__(self, cfg: GuideConfig, device: torch.device):
        super().__init__(cfg)
        # to device
        # self.pipe.enable_model_cpu_offload()  # to slow
        self.pipe.to(device)
        self.device = device
        # link each component
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(num_inference_steps=1000)
        self.tp_scheduler = self.setup_scheduler(self.scheduler, device)

    def get_text_embeds(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ) -> dict:
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=True,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        prompt_embeds_kwargs = {'prompt_embeds': prompt_embeds}
        negative_prompt_embeds_kwargs = {'negative_prompt_embeds': negative_prompt_embeds}
        return prompt_embeds, negative_prompt_embeds, \
            prompt_embeds_kwargs, negative_prompt_embeds_kwargs

    def _predict(self, latents_model_input, text_embeddings):
        return self.unet(
            latents_model_input,
            timestep=self.timestep,
            encoder_hidden_states=text_embeddings,
        ).sample


class ScoreDistillationXL(BasicScoreDistillation, AutoEncoderSDXL):
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    pipe: StableDiffusionXLPipeline
    scheduler: EulerDiscreteScheduler

    def __init__(self, cfg: GuideConfig, device: torch.device):
        super().__init__(cfg)
        # to device
        self.pipe.to(device)
        self.device = device
        # link each component
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        # self.scheduler.set_timesteps(num_inference_steps=1000)
        self.tp_scheduler = self.setup_scheduler(self.scheduler, device)

    def get_text_embeds(
        self,
        prompt: List[str],
        negative_prompt: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ) -> dict:
        # text_encoder_lora_scale = (
        #     cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        # )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )
        prompt_embeds_kwargs = {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }
        negative_prompt_embeds_kwargs = {
            'negative_prompt_embeds': negative_prompt_embeds,
            'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
        }
        return (prompt_embeds, pooled_prompt_embeds), (negative_prompt_embeds, negative_pooled_prompt_embeds), \
            prompt_embeds_kwargs, negative_prompt_embeds_kwargs
        # return {
        #     'prompt_embeds': prompt_embeds,
        #     'pooled_prompt_embeds': pooled_prompt_embeds,
        #     'negative_prompt_embeds': negative_prompt_embeds,
        #     'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
        # }
        # return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_sdxl_inputs(
            self,
            text_embeddings: Tuple[torch.Tensor],
            batch_size: int,
            latent_height: int,
            latent_width: int,
            original_size: Tuple[int, int] = None,
            target_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
        ):
        device = self.device

        height = latent_height * self.vae_scale_factor
        width = latent_width * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        prompt_embeds = text_embeddings[0].to(device)
        pooled_prompt_embeds = text_embeddings[1].to(device)

        add_text_embeds = pooled_prompt_embeds
        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device).repeat(batch_size, 1)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return prompt_embeds, added_cond_kwargs

    def _predict(self, latents_model_input: torch.Tensor, text_embeddings: Union[torch.Tensor, List]):

        batch_size, _, latent_height, latent_width = latents_model_input.shape

        prompt_embeds, added_cond_kwargs = self.prepare_sdxl_inputs(
            text_embeddings=text_embeddings,
            batch_size=batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
        )

        return self.unet(
            latents_model_input,
            timestep=self.timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
