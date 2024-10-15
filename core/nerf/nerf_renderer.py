import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice, random
from typing import Optional, Iterable
import numpy as np
from loguru import logger
import nvdiffrast.torch

from configs import NeRFConfig
from core.optim.loss.mesh_loss import normal_consistency, laplacian_smooth_loss

from .isosurface.dmtet import DMTet, load_tets
from .nerf_utils import get_rays, sample_pdf, custom_meshgrid, safe_normalize, near_far_from_bound


class _NeRFRenderer(nn.Module):
    def __init__(self, cfg: NeRFConfig):
        super().__init__()

        self.opt = cfg
        self.bound = cfg.bound
        self.cascade = 1 + math.ceil(math.log2(cfg.bound))
        self.grid_size = cfg.grid_size
        self.cuda_ray = cfg.cuda_ray
        self.min_near = cfg.min_near
        self.density_thresh = cfg.density_thresh
        self.bg_mode = cfg.bg_mode
        self.bg_radius = cfg.bg_radius if self.bg_mode == 'nerf' else 0.0
        self.rand_bg_prob = cfg.rand_bg_prob
        self.latent_mode = cfg.nerf_type in ('latent', 'latent_approx')
        if self.latent_mode:
            self.img_dims = 4
            self.additional_dim_size = 0 if cfg.nerf_type == 'latent_approx' else 1
            self.bg_colors = {
                'white': torch.tensor([2.1750,  1.4431, -0.0317, -1.1624]),
                'black': torch.tensor([-0.9952, -2.6023,  1.1155,  1.2966]),
                'gray': torch.tensor([0.9053, -0.7003,  0.5424,  0.1057]),
            }
        else:
            self.img_dims = 3
            self.additional_dim_size = 1 if cfg.nerf_type == 'latent_tune' else 0
            self.bg_colors = {
                'white': torch.tensor([1.0, 1.0, 1.0]),
                'black': torch.tensor([0.0, 0.0, 0.0]),
                'gray': torch.tensor([0.5, 0.5, 0.5]),
            }

        if self.cuda_ray:
            logger.info('Loading CUDA ray marching module (compiling might take a while)...')
            from .raymarching import rgb as raymarchingrgb
            from .raymarching import latent as raymarchinglatent
            # logger.info('\tDone.')
            self.raymarching = raymarchinglatent if self.latent_mode else raymarchingrgb

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-cfg.bound, -cfg.bound, -cfg.bound, cfg.bound, cfg.bound, cfg.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def build_extra_state(self, grid_size):
        # density grid
        density_grid = torch.zeros([self.cascade, grid_size ** 3]) # [CAS, H * H * H]
        density_bitfield = torch.zeros(self.cascade * grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0
        self.min_density = None
        self.max_density = None
        # step counter
        step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=None, random_sigmas:bool=False):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return

        if S is None:
            S = self.grid_size

        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = self.raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        if random_sigmas:
                            # sigmas += torch.rand_like(sigmas) - 0.5
                            sigmas += 1.0 * torch.exp(-(cas_xyzs ** 2).sum(-1) / (2 * 0.2 ** 2))
                        # assign 
                        tmp_grid[cas, indices] = sigmas

        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.min_density = torch.log(torch.min(self.density_grid[valid_mask])).clamp(-15., 15.).item()
        self.max_density = torch.log(torch.max(self.density_grid[valid_mask])).clamp(-15., 15.).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = self.raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', perturb=False, num_steps=64, upsample_steps=32, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_mode: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        # query density and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        if shading == 'albedo':
            rgbs = density_outputs['albedo']
        else:
            _, rgbs = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, self.img_dims) # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]

        # calculate depth
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)
        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs
        # results['sigmas'] = sigmas
        # results['rgbs'] = rgbs
        # results['alphas'] = alphas

        return results

    def run_staged(self, rays_o, rays_d, max_ray_batch=4096, **kwargs):
        B, N = rays_o.shape[:2]
        device = rays_o.device

        depth = torch.empty((B, N), device=device)
        image = torch.empty((B, N, self.img_dims), device=device)
        weights_sum = torch.empty((B, N), device=device)

        for b in range(B):
            head = 0
            while head < N:
                tail = min(head + max_ray_batch, N)
                results_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                depth[b:b+1, head:tail] = results_['depth']
                weights_sum[b:b+1, head:tail] = results_['weights_sum']
                image[b:b+1, head:tail] = results_['image']
                head += max_ray_batch

        results = {}
        results['depth'] = depth
        results['image'] = image
        results['weights_sum'] = weights_sum
        return results

    def run_cuda(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', perturb=False,
                 dt_gamma=0, max_steps=1024, T_thresh=1e-4, **kwargs):  # max_steps=1024
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = self.raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)
        
        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        results = {}
        xyzs, sigmas = None, None
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, ts, rays = self.raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                self.cascade, self.grid_size, nears, fars, perturb, dt_gamma, max_steps)
            # cuda_ray=True, wo_pretrained: xyzs.shape[0] = 152427163
            # cuda_ray=True, w_pretrained: xyzs.shape[0] = 4357172

            sigmas, rgbs = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = self.raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh)

            # weights normalization
            results['weights'] = weights

        else:
            # allocate outputs 
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, self.img_dims, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0

            while step < max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = self.raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound,
                     self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                self.raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

        image = image.reshape(*prefix, self.img_dims)
        depth = depth.reshape(*prefix)
        weights_sum = weights_sum.reshape(*prefix)
        mask = (nears < fars).reshape(*prefix)
        
        # [B, H*W, C]
        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask

        # [N, C]
        results['xyzs'] = xyzs
        results['sigmas'] = sigmas
        results['rgbs'] = rgbs
        return results

    def render(self, data, shading='albedo', bg_mode=None, staged=False, **kwargs):
        c2w = data['c2w']
        intrinsics = data['intrinsics']
        H = data['image_height']
        W = data['image_width']

        # rays_o, rays_d: [B, N, 3], assumes B == 1
        rays_output = get_rays(c2w=c2w, intrinsics=intrinsics, H=H, W=W)
        rays_o, rays_d = rays_output['rays_o'], rays_output['rays_d']

        if self.training:
            kwargs['perturb'] = True
        else:
            kwargs['perturb'] = False

        if self.dmtet:
            results = self.run_dmtet(rays_o, rays_d, shading=shading, **kwargs)
        elif self.cuda_ray:
            results = self.run_cuda(rays_o, rays_d, shading=shading, **kwargs)
        else:
            if staged and not self.cuda_ray:  # never stage when cuda_ray
                results = self.run_staged(rays_o, rays_d, shading=shading, **kwargs)
            else:
                results = self.run(rays_o, rays_d, shading=shading, **kwargs)

        # Reshape
        B = rays_o.shape[0]
        results['image'] = results['image'].reshape(B, H, W, -1)  # .permute(0, 3, 1, 2).contiguous()
        results['depth'] = results['depth'].reshape(B, H, W, 1)
        results['mask'] = results['mask'].reshape(B, H, W, 1)
        results['weights_sum'] = results['weights_sum'].reshape(B, H, W, 1)
        results['alpha'] = results['weights_sum']

        if shading == 'normal':
            results['image'] = results['image'][:, :, :, :3]
        else:
            # mix background color
            image_fg = results['image']
            image_bg = self.background(image_fg, bg_mode, rays_d)
            if image_bg is not None:
                weights_sum = results['weights_sum']
                # if self.training and self.opt.bg_suppress:
                # if self.opt.bg_suppress:
                #     bg_suppress_rate = None
                #     image_bg_mixed = None
                #     with torch.no_grad():
                #         image_delta = image_fg - weights_sum * image_bg
                #         l1_dist = image_delta.abs()
                #         l1_dist = (l1_dist * weights_sum).sum() / weights_sum.sum()  # [0.0, 1.0]
                #         if l1_dist.item() < self.opt.bg_suppress_dist:
                #             bg_suppress_rate = self.opt.bg_suppress_dist - l1_dist.item()  # [0.0, bg_suppress_dist]
                            
                #             # color_fg = image_fg.mean(dim=(0, 2, 3))  # [3, ]
                #             # color_bg = image_bg.mean(dim=(0, 2, 3))  # [3, ]
                #             # color_mix = color_bg - color_fg
                #             # color_mix = torch.rand(self.img_dims)  # [3, ]
                #             # image_bg_mixed = color_mix.to(image_bg).expand_as(image_bg)
                #     if bg_suppress_rate is not None:
                #         image_bg = image_bg * (1 - bg_suppress_rate) + image_bg_mixed * bg_suppress_rate

                results['image_bg'] = image_bg
                results['image_fg'] = image_fg
                
                if self.opt.detach_bg_weights_sum:
                    results['image'] = image_fg + (1 - weights_sum.detach()) * image_bg
                else:
                    results['image'] = image_fg + (1 - weights_sum) * image_bg

        return results


class _DMTetRenderer(_NeRFRenderer):
    def __init__(self, cfg: NeRFConfig):
        super().__init__(cfg)
        self.dmtet = self.opt.dmtet
        if self.dmtet:
            # options
            self.lock_geo = self.opt.lock_geo

            # load dmtet vertices
            tets = load_tets(self.opt.tet_grid_size)

            self.verts = - torch.tensor(tets['vertices'], dtype=torch.float32) * 2 # covers [-1, 1]
            self.indices  = torch.tensor(tets['indices'], dtype=torch.long)
            self.tet_scale = torch.tensor([1, 1, 1], dtype=torch.float32)
            self.dmtet_model = None

            # vert sdf and deform
            sdf = torch.nn.Parameter(torch.zeros_like(self.verts[..., 0]), requires_grad=True)
            self.register_parameter('sdf', sdf)
            deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
            self.register_parameter('deform', deform)

            edges = torch.tensor([0,1, 0,2, 0,3, 1,2, 1,3, 2,3], dtype=torch.long) # six edges for each tetrahedron.
            all_edges = self.indices[:,edges].reshape(-1, 2) # [M * 6, 2]
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

            self.glctx = nvdiffrast.torch.RasterizeCudaContext()

    def run_dmtet(self, rays_o, rays_d, mvp, H, W, light_d=None, ambient_ratio=1.0, shading='albedo', **kwargs):
        # mvp: [B, 4, 4]

        campos = rays_o[:, 0, :]  # only need one ray per batch

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(campos + torch.randn_like(campos)).view(-1, 1, 1, 3) # [B, 1, 1, 3]

        results = {}

        # get mesh
        sdf = self.sdf
        deform = torch.tanh(self.deform) / self.opt.tet_grid_size

        verts, faces = self.dmtet_model(self.verts + deform, sdf, self.indices)

        # get normals
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

        faces = faces.int()

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = safe_normalize(face_normals)

        vn = torch.zeros_like(verts)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

        # rasterization
        verts_clip = F.pad(verts, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1)  # [1, N, 4]
        verts_clip = torch.bmm(verts_clip, mvp.permute(0, 2, 1)).float()  # [B, N, 4]
        rast, rast_db = nvdiffrast.torch.rasterize(self.glctx, verts_clip, faces, (H, W))

        alpha = (rast[..., 3:] > 0).float()
        xyzs, _ = nvdiffrast.torch.interpolate(verts.unsqueeze(0), rast, faces) # [B, H, W, 3]
        normal, _ = nvdiffrast.torch.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = safe_normalize(normal)

        xyzs = xyzs.view(-1, 3)
        mask = (rast[..., 3:] > 0).view(-1).detach()

        # do the lighting here since we have normal from mesh now.
        albedo = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            masked_albedo = self.density(xyzs[mask])['albedo']
            albedo[mask] = masked_albedo.float()
        mask = mask.view(-1, H, W, 1)
        albedo = albedo.view(-1, H, W, 3)

        # these two modes lead to no parameters to optimize if using --lock_geo.
        if self.opt.lock_geo and shading in ['textureless', 'normal']:
            shading = 'lambertian'

        # shading
        if shading == 'albedo':
            color = albedo
        elif shading == 'textureless':
            lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
            color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif shading == 'normal':
            color = (normal + 1) / 2
        else: # 'lambertian'
            lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
            color = albedo * lambertian.unsqueeze(-1)

        color = nvdiffrast.torch.antialias(color, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
        alpha = nvdiffrast.torch.antialias(alpha, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 1]

        depth = rast[:, :, :, [2]] # [B, H, W]

        results['mask'] = mask
        results['depth'] = depth
        results['image'] = color
        results['weights_sum'] = alpha.squeeze(-1)

        if self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0:
            normal_image = nvdiffrast.torch.antialias((normal + 1) / 2, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
            results['normal_image'] = normal_image

        # regularizations
        if self.training:
            if self.opt.lambda_mesh_normal > 0:
                results['loss_normal'] = normal_consistency(face_normals, faces)
            if self.opt.lambda_mesh_laplacian > 0:
                results['loss_laplacian'] = laplacian_smooth_loss(verts, faces)

        return results

    @torch.no_grad()
    def init_tet(self, device, H, W, mesh=None):
        self.verts = self.verts.to(device)
        self.indices = self.indices.to(device)
        self.tet_scale = self.tet_scale.to(device)
        self.all_edges = self.all_edges.to(device)
        self.dmtet_model = DMTet(device)

        if H > 2048 or W > 2048:
            self.glctx = nvdiffrast.torch.RasterizeGLContext()

        if mesh is not None:
            # normalize mesh
            scale = 0.8 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
            center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
            mesh.vertices = (mesh.vertices - center) * scale

            # init scale
            # self.tet_scale = torch.from_numpy(np.abs(mesh.vertices).max(axis=0) + 1e-1).to(self.verts.dtype).cuda()
            self.tet_scale = torch.from_numpy(np.array([np.abs(mesh.vertices).max()]) + 1e-1).to(self.verts.dtype).cuda()
            self.verts = self.verts * self.tet_scale

            # init sdf
            import cubvh
            BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
            sdf, _, _ = BVH.signed_distance(self.verts, return_uvw=False, mode='watertight')
            sdf *= -10 # INNER is POSITIVE, also make it stronger
            self.sdf.data += sdf.to(self.sdf.data.dtype).clamp(-1, 1)
        else:
            if self.cuda_ray and self.mean_density > 0:
                density_thresh = min(self.mean_density, self.density_thresh)
            else:
                density_thresh = self.density_thresh

            if self.opt.density_activation == 'softplus':
                density_thresh = density_thresh * 25

            # init scale
            sigma = self.density(self.verts)['sigma'] # verts covers [-1, 1] now
            mask = sigma > density_thresh
            valid_verts = self.verts[mask]
            self.tet_scale = valid_verts.abs().amax(dim=0) + 1e-1
            self.verts = self.verts * self.tet_scale

            # init sigma
            sigma = self.density(self.verts)['sigma'] # new verts
            self.sdf.data += (sigma - density_thresh).clamp(-1, 1)

        print(f'[INFO] init dmtet: scale = {self.tet_scale}')


NeRFRenderer = _DMTetRenderer
