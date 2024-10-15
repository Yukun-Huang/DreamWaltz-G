from PIL import Image
import numpy as np
from typing import Callable, Optional, List
import torch
from torch import Tensor
import torch.nn.functional as F


def visualize(imgs, image_path):
    from einops import rearrange
    from torchvision import utils as vutils
    from imageio import imwrite

    imgs = rearrange(imgs, "N H W C -> N C H W", C=3)
    imgs = torch.from_numpy(imgs)
    pane = vutils.make_grid(imgs, padding=2, nrow=4)
    pane = rearrange(pane, "C H W -> H W C", C=3).numpy()
    imwrite(image_path, pane)


def tensor2numpy(tensor: Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return tensor


def tensor2image(
    tensor: Tensor,
    color_channel: int = 1,
) -> Image.Image | List[Image.Image]:
    """
    Converts a tensor to an image.

    Args:
        tensor (Tensor): The input tensor. Should have shape [B, H, W, C] or [B, C, H, W].
            Values should be in the range [0.0, 1.0].
        color_channel (int): The index of color channels in the output image. Should be 1 or 3.

    Returns:
        Image.Image: The converted image(s).

    """
    assert tensor.ndim == 4, f"invalid tensor shape: {tensor.shape}"
    
    if tensor.size(0) > 1:
        # If there are multiple images, convert each image separately
        images = []
        for i in range(tensor.size(0)):
            images.append(tensor2image(tensor[i:i+1], color_channel))
        return images

    color_dimension = tensor.size(color_channel)
    assert color_channel in (1, 3), f"invalid color channel: {color_channel}"
    assert color_dimension in (1, 3, 4), f"invalid color dimension: {color_dimension}"

    if color_channel == 1:
        # Convert channel-first format to channel-last format
        tensor = tensor.permute(0, 2, 3, 1).contiguous()

    tensor = tensor[0].detach().cpu().numpy()
    tensor = (tensor * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    if tensor.shape[-1] == 3:  # RGB
        image = Image.fromarray(tensor)
    elif tensor.shape[-1] == 4:  # RGBA
        image = Image.fromarray(tensor, mode="RGBA")
    elif tensor.shape[-1] == 1:  # Grayscale
        image = Image.fromarray(tensor[..., 0], mode="L")

    return image


# def tensor2image(
#     tensor: Tensor,
#     color_channel: int,
#     vae_decode: Callable = None,
# ) -> Image.Image:
#     """
#     Args:
#         tensor: Tensor, [B, H, W, C] or [B, C, H, W]
#         color_channel: int, 1 or 3
#         vae_decode: Callable, vae decoder
#     """
#     assert tensor.size(0) == 1 and tensor.ndim == 4, f"Invalid tensor shape: {tensor.shape}"

#     with torch.inference_mode():
#         if color_channel == 3:

#         if color_channel == 1:
#             if tensor.shape[color_channel] == 4:
#                 tensor = vae_decode(tensor)
#             tensor = tensor.permute(0, 2, 3, 1).contiguous()
#         elif color_channel == 3:
#             if tensor.shape[color_channel] == 4:
#                 tensor = tensor.permute(0, 3, 1, 2).contiguous()
#                 if tensor.shape[-2] > 128 or tensor.shape[-1] > 128:
#                     tensor = F.interpolate(tensor, (tensor.shape[-2]//8, tensor.shape[-1]//8), mode='area')
#                 tensor = vae_decode(tensor)
#                 tensor = tensor.permute(0, 2, 3, 1).contiguous()
#         else:
#             raise NotImplementedError

#     tensor = tensor[0].detach().cpu().numpy()
#     # ndarray to Image: [0, 1] -> [0, 255]
#     tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
#     if tensor.shape[-1] == 3:  # rgb
#         image = Image.fromarray(tensor)
#     elif tensor.shape[-1] == 1:  # gray
#         image = Image.fromarray(tensor[..., 0], mode="L")
#     else:
#         assert 0, image.shape
#     return image
