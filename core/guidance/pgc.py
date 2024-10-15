
"""
"Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping". ICLR 2024.

Usage:
pred_rgbs: torch.Tensor [B, C, H, W]
pred_rgbs.register_hook(_hook)

"""
import torch
from torch.cuda.amp import GradScaler
from configs import TrainConfig


def build_grad_hook_func(grad_clip:bool, grad_norm:bool, grad_clip_scale:float, scaler: GradScaler = None, mask = None):
    
    def _hook(grad: torch.Tensor):

        if grad_clip:
            if mask is not None:
                grad *= mask.expand_as(grad)
                grad_for_std = grad[mask.expand_as(grad) > 0.5]
            else:
                grad_for_std = grad
            grad_for_std = grad_for_std.nan_to_num(0.0, 0.0, 0.0)
            std = ((grad_for_std ** 2).sum() / grad_for_std.count_nonzero()) ** 0.5 * grad_clip_scale
            grad_new = grad.clamp(-std, std).nan_to_num(0.0)
        else:
            grad_new = grad
        
        # print(grad.min(), grad.max(), grad.shape)
        # print(grad_new.min(), grad_new.max(), std)
        # print()
        
        if grad_norm:
            grad_new = torch.nn.functional.normalize(grad_new, p=2, dim=(1, 2, 3))
            if scaler is not None and scaler._enabled:
                grad_new *= scaler._get_scale_async()
        
        return grad_new

    return _hook


def build_pgc_hook_func(clip_value: float, pgc_suppress_type: int, scaler: GradScaler = None):

    def _hook(grad: torch.Tensor):

        if scaler is not None and scaler._enabled:
            clip_value *= scaler._get_scale_async()

        if pgc_suppress_type == 0:  # Pixel-wise Clip
            ratio = 1. / grad.abs() * clip_value
            ratio[ratio > 1.0] = 1.0
            grad_ = grad * torch.amin(ratio, dim=[1], keepdim=True)
        
        elif pgc_suppress_type == 1:  # Clip
            grad_ = grad.clamp(-clip_value, clip_value)
        
        elif pgc_suppress_type == 2:  # Global Scale
            grad_ = grad / grad.abs().max() * clip_value
        
        elif pgc_suppress_type == 3:  # Sigmoid
            grad_ = (torch.sigmoid(grad) - 0.5) * clip_value
        
        elif pgc_suppress_type == 4:  # PNGD
            grad_norm = grad.abs()
            grad_ = clip_value * (grad / (grad_norm + clip_value))
        
        elif pgc_suppress_type == 5:  # PNGD
            grad_norm = torch.amax(grad.abs(), dim=[1], keepdim=True)
            grad_ = clip_value * (grad / (grad_norm + clip_value))
        
        else:
            grad_ = grad

        return grad_

    return _hook
