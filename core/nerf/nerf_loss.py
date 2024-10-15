import torch
from torch import nn
import numpy as np
from torch import Tensor
from igl import read_obj, fast_winding_number_for_meshes, point_mesh_squared_distance
from configs import NeRFConfig


# Orientation Loss
def orientation_loss(weights: Tensor, normals: Tensor, dirs: Tensor):
    # orientation loss 
    loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
    return loss_orient.mean()


# Sparsity Loss
def opacity_loss(pred_ws):
    loss_opacity = torch.sqrt((pred_ws ** 2 + 0.01).mean())
    return loss_opacity

def entropy_loss(pred_ws, eps=1e-6):
    alphas = pred_ws.clamp(eps, 1 - eps)
    entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
    return entropy

def emptiness_loss(pred_ws, emptiness_weight=10000, emptiness_scale=10):
    loss = torch.log(1 + emptiness_scale * pred_ws).mean()
    return emptiness_weight * loss

class SparsityLoss:
    def __init__(self, cfg: NeRFConfig, use_schedule: bool = True) -> None:
        self.lambda_opacity = cfg.lambda_opacity
        self.lambda_entropy = cfg.lambda_entropy
        self.lambda_emptiness = cfg.lambda_emptiness
        self.use_schedule = use_schedule
        self.sparsity_multiplier = cfg.sparsity_multiplier
        self.sparsity_step = cfg.sparsity_step
        self.available = self.lambda_opacity > 0.0 or self.lambda_entropy > 0.0 or self.lambda_emptiness > 0.0

    def __call__(self, pred_ws, current_step=None, max_iteration=None):

        loss = 0.0

        if self.lambda_opacity > 0.0:
            loss += self.lambda_opacity * opacity_loss(pred_ws)

        if self.lambda_entropy > 0.0:
            loss += self.lambda_entropy * entropy_loss(pred_ws)

        if self.lambda_emptiness > 0.0:
            loss += self.lambda_emptiness * emptiness_loss(pred_ws)

        if self.use_schedule and current_step / max_iteration >= self.sparsity_step:
            loss *= self.sparsity_multiplier

        return loss


# Shape Loss
class MeshOBJ:
    dx = torch.zeros(3).float()
    dx[0] = 1
    dy, dz = dx[[1, 0, 2]], dx[[2, 1, 0]]
    dx, dy, dz = dx[None, :], dy[None, :], dz[None, :]

    def __init__(self, v: np.ndarray, f: np.ndarray):
        self.v = v.astype(np.float32)
        self.f = f.astype(np.int64)
        self.dx, self.dy, self.dz = MeshOBJ.dx, MeshOBJ.dy, MeshOBJ.dz
        # self.v_tensor = torch.from_numpy(self.v)

        vf = self.v[self.f, :]
        self.f_center = vf.mean(axis=1)
        # self.f_center_tensor = torch.from_numpy(self.f_center).float()

        e1 = vf[:, 1, :] - vf[:, 0, :]
        e2 = vf[:, 2, :] - vf[:, 0, :]
        self.face_normals = np.cross(e1, e2)
        self.face_normals = self.face_normals / np.linalg.norm(self.face_normals, axis=-1)[:, None]
        # self.face_normals_tensor = torch.from_numpy(self.face_normals)

    def normalize_mesh(self, target_scale=0.5):
        verts = self.v

        # Compute center of bounding box
        # center = torch.mean(torch.column_stack([torch.max(verts, dim=0)[0], torch.min(verts, dim=0)[0]]))
        center = verts.mean(axis=0)
        verts = verts - center
        scale = np.max(np.linalg.norm(verts, axis=1))
        verts = (verts / scale) * target_scale

        return MeshOBJ(verts, self.f)

    def winding_number(self, query: Tensor) -> Tensor:
        device = query.device
        shp = query.shape
        query_np = query.detach().cpu().reshape(-1, 3).numpy()
        target_alphas = fast_winding_number_for_meshes(self.v, self.f, query_np)
        return torch.from_numpy(target_alphas).reshape(shp[:-1]).to(device)

    def gaussian_weighted_distance(self, query: Tensor, sigma: Tensor) -> Tensor:
        device = query.device
        shp = query.shape
        query_np = query.detach().cpu().reshape(-1, 3).numpy()
        distances, _, _ = point_mesh_squared_distance(query_np, self.v, self.f)
        distances = torch.from_numpy(distances).reshape(shp[:-1]).to(device)
        weight = torch.exp(-(distances / (2 * sigma**2)))
        return weight

def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.01):
        return v.clamp(T, 1 - T)
    ce = -1 * (p * torch.log(clamp(q)) + (1 - p) * torch.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()

class ShapeLoss(nn.Module):
    def __init__(self, shape_path:str, mesh_scale:float=0.7):
        super().__init__()
        v, _, _, f, _, _ = read_obj(shape_path, float)
        mesh = MeshOBJ(v, f)
        self.sketchshape = mesh.normalize_mesh(mesh_scale)

    def forward(self, xyzs, sigmas, delta:float=0.2, proximal_surface:float=0.3):
        mesh_occ = self.sketchshape.winding_number(xyzs)
        if proximal_surface > 0:
            weight = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, proximal_surface)
        else:
            weight = None
        indicator = (mesh_occ > 0.5).float()
        nerf_occ = 1 - torch.exp(-delta * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(nerf_occ, indicator, weight=weight)  # order is important for CE loss + second argument may not be optimized
        return loss

class DynamicShapeLoss(nn.Module):
    def __init__(self, cfg: NeRFConfig):
        super().__init__()
        self.lambda_shape = cfg.lambda_shape

    def forward(self, xyzs: Tensor, sigmas: Tensor, v: np.ndarray, f: np.ndarray, delta:float=0.2, proximal_surface:float=0.1):
        mesh = MeshOBJ(v, f)
        mesh_occ = mesh.winding_number(xyzs)
        if proximal_surface > 0:
            # weight = 1 - mesh.gaussian_weighted_distance(xyzs, proximal_surface)
            weight = mesh.gaussian_weighted_distance(xyzs, proximal_surface)
        else:
            weight = None
        indicator = (mesh_occ > 0.5).float()
        nerf_occ = 1 - torch.exp(-delta * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(nerf_occ, indicator, weight=weight)  # order is important for CE loss + second argument may not be optimized
        return loss * self.lambda_shape
