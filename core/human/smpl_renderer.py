import os.path as osp
import torch
import torch.nn as nn
from typing import Union, Dict
from smplx.body_models import SMPLXOutput
from configs.paths import HUMAN_TEMPLATES as MODEL_ROOT
from utils.mesh import Mesh, MeshRenderer, vertex_colors_to_albedo_image
from .smpl_model import SMPLModel


def trunc_rev_sigmoid(x: torch.Tensor, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


class SMPLRenderer(nn.Module):
    def __init__(self, smpl_model: SMPLModel, body_mapping=None, albedo='smplx', albedo_res=1024) -> None:
        super().__init__()

        self.device = smpl_model.device
        self.model_type = smpl_model.model_type
        self.albedo_res = albedo_res

        # init template mesh
        mesh = self.load_uv_data()
        self.register_buffer('f', mesh.f)  # [20908, 3]
        self.register_buffer('ft', mesh.ft)  # [20908, 3]
        self.register_buffer('vt', mesh.vt)  # [11313, 2]

        # init albedo
        albedo = self.get_init_albedo(albedo, albedo_res, body_mapping=body_mapping, mesh=mesh)
        self.albedo = nn.Parameter(albedo)

        # init renderer
        self.renderer = MeshRenderer()

    def get_init_albedo(self, albedo, albedo_res, body_mapping=None, mesh=None):
        if albedo == 'zeros':
            albedo = torch.zeros(
                (albedo_res, albedo_res, 3),
                dtype=torch.float32,
                device=self.device,
            )
        elif albedo == 'ones':
            albedo = torch.ones(
                (albedo_res, albedo_res, 3),
                dtype=torch.float32,
                device=self.device,
            )
        elif albedo == 'smplx':
            albedo = mesh.albedo
            # mesh.load_albedo(albedo_path)
        elif body_mapping is not None:
            skin_vids, skin_color = body_mapping['skin']['vertex_indices'], body_mapping['skin']['rgb']
            eyes_vids, eyes_color = body_mapping['eyes']['vertex_indices'], body_mapping['eyes']['rgb']
            vertex_colors = torch.zeros_like(mesh.v)  # [V, 3]
            vertex_colors[skin_vids] = torch.tensor([skin_color], device=self.device)
            vertex_colors[eyes_vids] = torch.tensor([eyes_color], device=self.device)
            albedo, albedo_mask = vertex_colors_to_albedo_image(
                vertex_colors=vertex_colors, f=mesh.f, vt=mesh.vt, ft=mesh.ft,
                h=albedo_res, w=albedo_res, save_path='albedo.png',
            )
        return trunc_rev_sigmoid(albedo)

    def load_uv_data(self) -> Mesh:
        if self.model_type == 'smplx':
            obj_path = osp.join(MODEL_ROOT, 'smplx/uv_map/smplx_uv.obj')
            albedo_path = osp.join(MODEL_ROOT, 'smplx/uv_map/smplx_uv.png')
            return Mesh.load_obj(
                obj_path,
                device=self.device,
                init_empty_tex=False,
                albedo_path=albedo_path,
                albedo_res=self.albedo_res,
            )
        else:
            assert 0, self.model_type

    def get_mesh(self, smpl_outputs: SMPLXOutput) -> Mesh:
        v = smpl_outputs.vertices
        assert v.shape[0] == 1
        mesh = Mesh(
            v=v[0],  # assert batch size = 1
            f=self.f,
            vt=self.vt,
            ft=self.ft,
            device=self.device,
        )
        mesh.auto_normal()
        mesh.set_albedo(self.albedo)
        return mesh

    def render(self, smpl_outputs: SMPLXOutput, data: dict, mesh_only=True, shading='albedo'):
        """
        Returns:
            image: torch.Tensor, [B, H, W, 3]
            normal: torch.Tensor, [B, H, W, 3]
            depth: torch.Tensor, [B, H, W, 1]
            alpha: torch.Tensor, [B, H, W, 1]
        """
        mesh = self.get_mesh(smpl_outputs)
        if mesh_only:
            return {'mesh': mesh}
        render_outputs = self.renderer.forward(
            mesh=mesh,
            data=data,
            shading=shading,
        )
        render_outputs['mesh'] = mesh
        return render_outputs
