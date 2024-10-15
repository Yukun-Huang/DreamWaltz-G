import torch
from loguru import logger
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from configs import GuideConfig
from .basic import build_vae, BasicStableDiffusion


class BasicAutoEncoder:
    vae: AutoencoderKL

    def __init__(self, cfg: GuideConfig, device=None):
        if not isinstance(self, BasicStableDiffusion):
            class_name = self.__class__.__name__
            self.vae = build_vae(
                model_name=cfg.diffusion,
                torch_dtype=torch.float16 if cfg.diffusion_fp16 else torch.float32,
                use_sdxl=cfg.use_sdxl,
            )
            if device is not None:
                self.vae = self.vae.to(device)
            logger.info(f'\t Successfully loaded {class_name}!')
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


class AutoEncoderSD(BasicAutoEncoder):
    def encode_images(self, images):
        if isinstance(images, torch.Tensor):
            images = self.image_processor.normalize(images)
        else:
            images = self.image_processor.preprocess(images).to(self.vae.device)
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents, output_type='pt'):
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        images = self.vae.decode(1 / self.vae.config.scaling_factor * latents, return_dict=False)[0]
        images = self.image_processor.postprocess(images.float(), output_type=output_type)
        return images


class AutoEncoderSDXL(BasicAutoEncoder):
    def encode_images(self, images):
        if isinstance(images, torch.Tensor):
            images = self.image_processor.normalize(images)
        else:
            images = self.image_processor.preprocess(images).to(self.vae.device)
        images = images.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def decode_latents(self, latents, output_type='pt'):
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        images = self.vae.decode(1 / self.vae.config.scaling_factor * latents, return_dict=False)[0]
        images = self.image_processor.postprocess(images.float(), output_type=output_type)
        return images

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)


# class AutoEncoderSDXL(BasicAutoEncoder):
#     def encode_images(self, images):
#         if isinstance(images, torch.Tensor):
#             images = self.image_processor.normalize(images)
#         else:
#             images = self.image_processor.preprocess(images).to(self.vae.device)
#         dtype_images, dtype_vae = images.dtype, self.vae.dtype
#         if self.vae.config.force_upcast:
#             images = images.float()
#             self.vae.to(dtype=torch.float32)
#         with torch.cuda.amp.autocast(enabled=False):
#             latents = self.vae.encode(images).latent_dist.sample()
#         if self.vae.config.force_upcast:
#             latents = latents.to(dtype_images)
#             self.vae.to(dtype_vae)
#         return latents * self.vae.config.scaling_factor

#     def decode_latents(self, latents, output_type='pt'):
#         # make sure the VAE is in float32 mode, as it overflows in float16
#         needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
#         if needs_upcasting:
#             self.upcast_vae()
#         latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
#         with torch.cuda.amp.autocast(enabled=False):
#             images = self.vae.decode(1 / self.vae.config.scaling_factor * latents, return_dict=False)[0]
#         # cast back to fp16 if needed
#         if needs_upcasting:
#             self.vae.to(dtype=torch.float16)
#         images = self.image_processor.postprocess(images.float(), output_type=output_type)
#         return images

#     def upcast_vae(self):
#         dtype = self.vae.dtype
#         self.vae.to(dtype=torch.float32)
#         use_torch_2_0_or_xformers = isinstance(
#             self.vae.decoder.mid_block.attentions[0].processor,
#             (
#                 AttnProcessor2_0,
#                 XFormersAttnProcessor,
#                 LoRAXFormersAttnProcessor,
#                 LoRAAttnProcessor2_0,
#             ),
#         )
#         # if xformers or torch_2_0 is used attention block does not need
#         # to be in float32 which can save lots of memory
#         if use_torch_2_0_or_xformers:
#             self.vae.post_quant_conv.to(dtype)
#             self.vae.decoder.conv_in.to(dtype)
#             self.vae.decoder.mid_block.to(dtype)
