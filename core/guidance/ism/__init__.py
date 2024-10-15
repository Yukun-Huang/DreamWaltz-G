import torch
import math
from .guidance.sd_utils import StableDiffusion


class GuidanceParams():
    def __init__(self):
        self.guidance = "SD"        
        self.g_device = "cuda"

        self.model_key = 'stabilityai/stable-diffusion-2-1-base'
        self.is_safe_tensor = False
        self.base_model_key = None

        self.controlnet_model_key = None

        self.perpneg = True
        self.negative_w = -2.
        self.front_decay_factor = 2.
        self.side_decay_factor = 10.   
        
        self.vram_O = False
        self.fp16 = True
        self.hf_key = None
        self.t_range = [0.02, 0.5]     
        self.max_t_range = 0.98
        
        self.scheduler_type = 'DDIM'
        self.num_train_timesteps = None 

        self.sds = False
        self.fix_noise = False
        self.noise_seed = 0

        self.ddim_inv = False
        self.delta_t = 80
        self.delta_t_start = 100
        self.annealing_intervals = True
        self.text = ''
        self.inverse_text = ''
        self.textual_inversion_path = None
        self.LoRA_path = None
        self.controlnet_ratio = 0.5
        self.negative = ""
        self.guidance_scale = 7.5
        self.denoise_guidance_scale = 1.0
        self.lambda_guidance = 1.

        self.xs_delta_t = 200
        self.xs_inv_steps = 5
        self.xs_eta = 0.0

        # multi-batch
        self.C_batch_size = 1

        self.vis_interval = 100


def adjust_text_embeddings(embeddings, azimuth, guidance_opt):
    #TODO: add prenerg functions
    text_z_list = []
    weights_list = []
    K = 0
    #for b in range(azimuth):
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)
    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)


def prepare_embeddings(guidance_opt, guidance: StableDiffusion):
    embeddings = {}
    # text embeddings (stable-diffusion) and (IF)
    embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    embeddings['uncond'] = guidance.get_text_embeds([guidance_opt.negative])

    for d in ['front', 'side', 'back']:
        embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
    embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text)
    return embeddings


def guidance_setup(guidance_opt):
    if guidance_opt.guidance=="SD":
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O, 
                                   guidance_opt.t_range, guidance_opt.max_t_range, 
                                   num_train_timesteps=guidance_opt.num_train_timesteps, 
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path = guidance_opt.textual_inversion_path,
                                   LoRA_path = guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance)
    return guidance, embeddings
