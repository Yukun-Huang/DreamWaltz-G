import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr


def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[
        1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def compute_normal(vertices, faces, device=None):
    """
    Args:
        vertices: torch.Tensor, [V, 3] 或 [B, V, 3] 顶点坐标
        faces: torch.Tensor, [N, 3] 面片顶点索引
        device: str, optional, 使用的计算设备 (CPU 或 GPU)
    
    Returns:
        vertex_normals: torch.Tensor, [V, 3] 或 [B, V, 3] 顶点法线
        face_normals: torch.Tensor, [B, N, 3] 面片法线
    """
    if device is None:
        device = vertices.device
    
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices, dtype=torch.float, device=device)
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(faces.astype(np.int64), dtype=torch.long, device=device)
    
    vertices = vertices.to(device)
    faces = faces.to(device)

    # 如果输入是 [V, 3]，则在第0维添加一个维度变为 [1, V, 3]
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)

    B = vertices.shape[0]  # 批次大小
    V = vertices.shape[1]  # 顶点数
    N = faces.shape[0]     # 面片数

    # 提取每个面片的三个顶点索引
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()

    # 根据索引获取顶点坐标，扩展到批次维度
    v0 = vertices[:, i0, :]
    v1 = vertices[:, i1, :]
    v2 = vertices[:, i2, :]

    # 计算面法线
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = safe_normalize(face_normals)

    # 累加面法线到顶点
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.scatter_add_(1, i0[None, :, None].expand(B, -1, 3), face_normals)
    vertex_normals.scatter_add_(1, i1[None, :, None].expand(B, -1, 3), face_normals)
    vertex_normals.scatter_add_(1, i2[None, :, None].expand(B, -1, 3), face_normals)

    # 对顶点法线进行归一化，处理退化的法线向量
    vertex_normals = torch.where(
        dot(vertex_normals, vertex_normals) > 1e-20,
        vertex_normals,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    )
    vertex_normals = safe_normalize(vertex_normals)

    # 如果原始输入是 [V, 3]，则将结果还原为 [V, 3]
    if vertices.shape[0] == 1:
        vertex_normals = vertex_normals.squeeze(0)
        face_normals = face_normals.squeeze(0)
    
    return vertex_normals, face_normals


# def compute_normal(vertices, faces, device=None):
#     """
#     Args:
#         vertices: torch.Tensor, [V, 3]
#         faces: torch.Tensor, [N, 3]
#     """
#     if device is None:
#         device = vertices.device
    
#     if not isinstance(vertices, torch.Tensor):
#         vertices = torch.as_tensor(vertices, dtype=torch.float, device=device)
#     if not isinstance(faces, torch.Tensor):
#         faces = torch.as_tensor(faces.astype(np.int64), dtype=torch.long, device=device)
    
#     vertices = vertices.to(device)
#     faces = faces.to(device)

#     i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()

#     v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

#     face_normals = torch.cross(v1 - v0, v2 - v0)
#     face_normals = safe_normalize(face_normals)

#     # Splat face normals to vertices
#     vertex_normals = torch.zeros_like(vertices)
#     vertex_normals.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
#     vertex_normals.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
#     vertex_normals.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

#     # Normalize, replace zero (degenerated) normals with some default value
#     vertex_normals = torch.where(
#         dot(vertex_normals, vertex_normals) > 1e-20,
#         vertex_normals,
#         torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
#         # torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
#     )
#     vertex_normals = safe_normalize(vertex_normals)
    
#     return vertex_normals, face_normals



def export_normal_nvdiffrast(
    v: torch.Tensor,
    f: torch.Tensor,
    mvp: torch.Tensor,
    h: int,
    w: int,
    glctx,
    black_background: bool = True,
    normal_in_world: bool = True,
    same_scene: bool = False,
) -> torch.Tensor:
    """
    计算并导出三角形网格模型的法线图，支持批量处理，并可选择将多个批次渲染到同一张图像上。

    Args:
        v (torch.Tensor): 顶点坐标，形状为 [V, 3] 或 [B, V, 3]。
        f (torch.Tensor): 面片索引，形状为 [F, 3]。
        mvp (torch.Tensor): 模型-视图-投影矩阵，形状为 [B, 4, 4] 或 [1, 4, 4]。
        h (int): 输出图像的高度。
        w (int): 输出图像的宽度。
        glctx: OpenGL 上下文，用于光栅化操作。
        black_background (bool, optional): 是否使用黑色背景。默认为 True。
        normal_in_world (bool, optional): 是否在世界坐标系下计算法线。默认为 True。
        same_scene (bool, optional): 是否将多个批次渲染到同一张图像上。默认为 False。

    Returns:
        torch.Tensor: 法线图，形状为 [B, H, W, 3] 或 [H, W, 3]（取决于输入的顶点格式和 same_scene 参数）。
    """
    # 如果输入是 [V, 3]，则扩展维度变为 [1, V, 3]，以便统一处理批量输入
    if v.dim() == 2:
        v = v.unsqueeze(0)

    B = v.shape[0]  # 批次大小
    V = v.shape[1]  # 顶点数

    # 如果 normal_in_world 为 True，则在世界坐标系下计算法线
    if normal_in_world:
        vn, _ = compute_normal(v, f)  # 计算顶点法线，compute_normal 支持 [B, V, 3] 格式

    # 将顶点坐标转换为齐次坐标，并根据批次进行扩展
    v_homo = F.pad(v, pad=(0, 1), mode='constant', value=1.0)  # [B, V, 4]

    # 如果 mvp 的批次维度为 1，但 v_homo 的批次大于 1，则扩展 mvp
    if mvp.shape[0] == 1 and B > 1:
        mvp = mvp.expand(B, -1, -1)

    # 进行 MVP 变换，得到裁剪坐标系下的顶点位置
    v_clip = torch.bmm(v_homo, torch.transpose(mvp, 1, 2)).float()  # [B, V, 4]

    if same_scene:
        # 如果启用 same_scene 模式，将多个批次的顶点合并到一起
        v_clip = v_clip.reshape(-1, 4)  # [B*V, 4]
        vn = vn.reshape(-1, 3)          # [B*V, 3]

        # 处理面片索引，将其偏移以适应展开后的顶点
        f_offset = torch.arange(B, device=f.device, dtype=torch.long).view(B, 1, 1) * V  # 每个批次的偏移量
        f = (f.unsqueeze(0) + f_offset).view(-1, 3)  # [B * F, 3]
        f = f.to(dtype=torch.int32)

        # 光栅化操作：合并后的顶点和面片一起渲染到同一张图像上
        res = (h, w)
        rast, rast_db = dr.rasterize(glctx, v_clip[None, ...], f, res)  # 注意扩展维度以适应光栅化接口

        # 插值计算法线
        normal, _ = dr.interpolate(vn[None, ...].float(), rast, f)

        # 输出法线图形状为 [H, W, 3]
        normal = (normal + 1) / 2.

        if black_background:
            mask = rast[..., [3]] > 0.
            normal *= mask

        normal = dr.antialias(normal, rast, v_clip[None, ...], f).clamp(0, 1)
        normal = normal.squeeze(0)  # 移除多余的批次维度

    else:
        # 否则，正常处理每个批次，分别渲染成单独的图像
        res = (h, w)
        rast, rast_db = dr.rasterize(glctx, v_clip, f, res)

        # 如果不在世界坐标系下计算法线，则在裁剪坐标系中计算法线
        if not normal_in_world:
            vn, _ = compute_normal(v_clip[..., :3], f)

        # 插值计算法线
        normal, _ = dr.interpolate(vn, rast, f)  # [B, H, W, 3]

        # 归一化法线到 [0, 1] 范围
        normal = (normal + 1) / 2.

        if black_background:
            mask = rast[..., [3]] > 0.
            normal *= mask

        # 执行抗锯齿处理，并将结果限制在 [0, 1] 范围内
        normal = dr.antialias(normal, rast, v_clip, f).clamp(0, 1)  # [B, H, W, 3]

        # 如果输入是 [V, 3]，则去除批次维度以与原始输入格式匹配
        if v.shape[0] == 1:
            normal = normal.squeeze(0)

    return normal


# def export_normal_nvdiffrast(v, f, mvp, h, w, glctx, black_background=True, normal_in_world=True):
#     """
#     Args:
#         v: vertices, torch.Tensor, [V, 3]
#         f: faces, torch.Tensor, [F, 3]
#         mvp: torch.Tensor, [B, 4, 4]
#     """
#     B = mvp.shape[0]
#     if normal_in_world:
#         vn, _ = compute_normal(v, f)
    
#     v_homo = F.pad(v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1)  # [B, V, 4]
#     v_clip = torch.bmm(v_homo, torch.transpose(mvp, 1, 2)).float()  # [B, V, 4]

#     res = (h, w)
#     rast, rast_db = dr.rasterize(glctx, v_clip, f, res)

#     if not normal_in_world:
#         vn, _ = compute_normal(v_clip[0, :, :3], f)
    
#     normal, _ = dr.interpolate(vn[None, ...].float(), rast, f)

#     normal = (normal + 1) / 2.

#     if black_background:
#         mask = rast[..., [3]] > 0.
#         normal *= mask

#     normal = dr.antialias(normal, rast, v_clip, f).clamp(0, 1)  # [B, H, W, 3]

#     return normal


class Mesh:
    def __init__(self, v=None, f=None, vn=None, fn=None, vt=None, ft=None, albedo=None, device=None, base=None,
                 init_empty_tex=False, albedo_res=1024):
        """
        Args:
            v: vertices, torch.Tensor, [V, 3]
            f: faces, torch.Tensor, [F, 3]
        """
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft
        self.v_tng = None
        self.f_tng = None
        # only support a single albedo
        if init_empty_tex:
            self.albedo = torch.zeros((albedo_res, albedo_res, 3), dtype=torch.float32, device=device)
        else:
            self.albedo = albedo
        self.device = device

        if isinstance(base, Mesh):
            for name in ['v', 'vn', 'vt', 'f', 'fn', 'ft', 'albedo', 'v_tng', 'f_tng']:
                if getattr(self, name) is None:
                    setattr(self, name, getattr(base, name))

    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None, init_empty_tex=False, albedo_res=1024,
                 uv_path=None, normalize=False):

        assert os.path.splitext(path)[-1] == '.obj'

        mesh = cls()

        # device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mesh.device = device

        # try to find texture from mtl file
        if albedo_path is None:
            mtl_path = path.replace('.obj', '.mtl')
            if os.path.exists(mtl_path):
                with open(mtl_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0: continue
                    prefix = split_line[0]
                    # NOTE: simply use the first map_Kd as albedo!
                    if 'map_Kd' in prefix:
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f'[load_obj] use albedo from: {albedo_path}')
                        break

        if init_empty_tex or albedo_path is None or not os.path.exists(albedo_path):
            # init an empty texture
            print(f'[load_obj] init empty albedo!')
            albedo = np.ones((albedo_res, albedo_res, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color
        else:
            albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
            albedo = cv2.resize(albedo, (albedo_res, albedo_res))
            albedo = albedo.astype(np.float32) / 255

        mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)

        # load obj
        with open(path, 'r') as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != '' else -1 for x in fv.split('/')]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0: continue
            # v/vn/vt
            prefix = split_line[0].lower()
            if prefix == 'v':
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == 'vn':
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == 'vt':
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == 'f':
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        mesh.vt = torch.tensor(texcoords, dtype=torch.float32, device=device) if len(texcoords) > 0 else None
        mesh.vn = torch.tensor(normals, dtype=torch.float32, device=device) if len(normals) > 0 else None

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = torch.tensor(tfaces, dtype=torch.int32, device=device) if texcoords is not None else None
        mesh.fn = torch.tensor(nfaces, dtype=torch.int32, device=device) if normals is not None else None

        # auto-normalize
        # Skip this
        if normalize:
            mesh.auto_size()

        print(f'[load_obj] v: {mesh.v.shape}, f: {mesh.f.shape}')

        # auto-fix normal
        if mesh.vn is None:
            mesh.auto_normal()

        print(f'[load_obj] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}')

        # auto-fix texture
        if mesh.vt is None:
            mesh.auto_uv(cache_path=uv_path)

        print(f'[load_obj] vt: {mesh.vt.shape}, ft: {mesh.ft.shape}')

        return mesh

    @classmethod
    def load_albedo(cls, albedo_path):
        albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        albedo = albedo.astype(np.float32) / 255
        return albedo

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self):  # to [-0.5, 0.5]
        vmin, vmax = self.aabb()
        scale = 1 / torch.max(vmax - vmin).item()
        self.v = self.v - (vmax + vmin) / 2  # Center mesh on origin
        self.v = self.v * scale

    def auto_normal(self):
        self.vn, self.fn = compute_normal(self.v, self.f, self.device)

    @torch.no_grad()
    def auto_uv(self, cache_path="", v=None, f=None):
        # try to load cache
        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np = data['vt'], data['ft']
        else:
            import xatlas
            if v is not None and f is not None:
                v_np = v.cpu().numpy()
                f_np = f.int().cpu().numpy()
            else:
                v_np = self.v.cpu().numpy()
                f_np = self.f.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        # save to cache
        # np.savez(cache_path, vt=vt_np, ft=ft_np)

        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)

        self.vt = vt
        self.ft = ft
        return vt, ft

    def compute_tangents(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v[self.f[:, i]]
            tex[i] = self.vt[self.ft[:, i]]
            vn_idx[i] = self.fn[:, i]

        tangents = torch.zeros_like(self.vn)
        tansum = torch.zeros_like(self.vn)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
        denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = safe_normalize(tangents)
        tangents = safe_normalize(tangents - dot(tangents, self.vn) * self.vn)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))
        self.v_tng = tangents
        self.f_tng = self.fn

    # write to obj file
    def write(self, path):

        mtl_path = path.replace('.obj', '.mtl')
        albedo_path = path.replace('.obj', '_albedo.png')

        v_np = self.v.cpu().numpy()
        vt_np = self.vt.cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.cpu().numpy() if self.vn is not None else None
        f_np = self.f.cpu().numpy()
        ft_np = self.ft.cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.cpu().numpy() if self.fn is not None else None

        with open(path, "w") as fp:
            fp.write(f'mtllib {os.path.basename(mtl_path)} \n')

            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            for v in vt_np:
                fp.write(f'vt {v[0]} {1 - v[1]} \n')

            for v in vn_np:
                fp.write(f'vn {v[0]} {v[1]} {v[2]} \n')

            fp.write(f'usemtl defaultMat \n')
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n')

        with open(mtl_path, "w") as fp:
            fp.write(f'newmtl defaultMat \n')
            fp.write(f'Ka 1 1 1 \n')
            fp.write(f'Kd 1 1 1 \n')
            fp.write(f'Ks 0 0 0 \n')
            fp.write(f'Tr 1 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0 \n')
            fp.write(f'map_Kd {os.path.basename(albedo_path)} \n')

        albedo = self.albedo.detach().cpu().numpy()
        albedo = (albedo * 255).astype(np.uint8)
        cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))

    def set_albedo(self, albedo):
        self.albedo = torch.sigmoid(albedo)

    def set_uv(self, vt, ft):
        self.vt = vt
        self.ft = ft

    def auto_uv_face_att(self):
        import kaolin as kal
        self.uv_face_att = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long(),
        )


class MeshRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.glctx = dr.RasterizeCudaContext()
        except:
            self.glctx = dr.RasterizeGLContext()

    def get_mlp_texture(self, mesh, mlp_texture, rast, rast_db, res=2048):
        # uv = mesh.vt[None, ...] * 2.0 - 1.0
        uv = mesh.vt[None, ...]

        # pad to four component coordinate
        uv4 = torch.cat((uv, torch.zeros_like(uv[..., 0:1]), torch.ones_like(uv[..., 0:1])), dim=-1)

        # rasterize
        _rast, _ = dr.rasterize(self.glctx, uv4, mesh.f.int(), (res, res))
        # print("_rast ", _rast.shape)
        # Interpolate world space position
        # gb_pos, _ = dr.interpolate(mesh.v[None, ...], _rast, mesh.f.int())

        # Sample out textures from MLP
        tex = mlp_texture.sample(_rast[..., :-1].view(-1, 3)).view(*_rast.shape[:-1], 3)

        texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all')
        # print(tex.shape)

        albedo = dr.texture(
            tex, texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
        # albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background

        # print(tex.shape, albedo.shape)
        # exit()
        return albedo

    @staticmethod
    def get_2d_texture(mesh, rast, rast_db):
        texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all')
        # print(texc.shape)
        albedo = dr.texture(mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
        albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background
        return albedo

    def forward(self, mesh: Mesh, data: dict,
            light_d=None, ambient_ratio=1., shading='albedo',
            spp=1, mlp_texture=None, is_train=False,
        ):
        """
        Args:
            spp:
            return_normal:
            transform_nml:
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            ambient_ratio: float
            shading: str shading type albedo, normal,
            ssp: int
        Returns:
            color:  [batch, h, w, 3]
            normal: [batch, h, w, 3]
            alpha:  [batch, h, w, 1]
            depth:  [batch, h, w, 1]
        """
        mvp, h, w = data['mvp'], data['image_height'], data['image_width']
        z_near, z_far = data['z_near'], data['z_far']

        B = mvp.shape[0]
        v_homo = F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1)  # [B, V, 4]
        v_clip = torch.bmm(v_homo, torch.transpose(mvp, 1, 2)).float()  # [B, V, 4]

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        # print(rast_db)

        ################################################################################
        # Interpolate attributes
        ################################################################################
        mask = rast[..., [3]] > 0.

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]
        depth, _ = dr.interpolate(v_clip[:, :, [3]], rast, mesh.f)  # [B, H, W, 1]
        depth_from_ndc = 2 * z_far * z_near / (rast[..., [2]] * (z_near - z_far) + z_near + z_far)  # [B, H, W, 1]

        if is_train:
            vn, _ = compute_normal(v_clip[0, :, :3], mesh.f)
            normal, _ = dr.interpolate(vn[None, ...].float(), rast, mesh.f)
        else:
            normal, _ = dr.interpolate(mesh.vn[None, ...].float(), rast, mesh.f)

        # Texture coordinate
        if not shading == 'normal':
            if mlp_texture is not None:
                albedo = self.get_mlp_texture(mesh, mlp_texture, rast, rast_db)
            else:
                albedo = self.get_2d_texture(mesh, rast, rast_db)

        if shading == 'normal':
            color = (normal + 1) / 2.
        elif shading == 'albedo':
            color = albedo
        else:  # lambertian
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = albedo * lambertian.repeat(1, 1, 1, 3)

        normal = (normal + 1) / 2.

        normal = dr.antialias(normal, rast, v_clip, mesh.f).clamp(0, 1)  # [B, H, W, 3]
        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [B, H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [B, H, W, 1]
        # depth = dr.antialias(depth, rast, v_clip, mesh.f)  # [B, H, W, 1]

        depth_from_ndc = depth_from_ndc * mask

        # inverse super-sampling
        if spp > 1:
            color = scale_img_nhwc(color, (h, w))
            normal = scale_img_nhwc(normal, (h, w))
            alpha = scale_img_nhwc(alpha, (h, w))
            depth = scale_img_nhwc(depth, (h, w))

        return {
            'image': color,
            'normal': normal,
            'alpha': alpha,
            'mask': mask,
            'depth': depth,
            'depth_from_ndc': depth_from_ndc,
            'rast': rast,
            'rast_db': rast_db,
        }


def vertex_colors_to_albedo_image(
        vertex_colors: torch.Tensor, f: torch.Tensor, vt: torch.Tensor, ft: torch.Tensor, save_path:str=None,
        h: int = 1024, w: int = 1024, to_torch: bool = True,
    ):
    """
    Returns:
        if to_torch:
            albedo: torch.Tensor, dtype=float, shape=[H, W, 3]
            mask: torch.Tensor, dtype=float, shape=[H, W, 1]
        else:
            albedo: np.ndarray, dtype=uint8, shape=[H, W, 3]
            mask: np.ndarray, dtype=uint8, shape=[H, W, 1]
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    device = vertex_colors.device
    
    # render uv maps
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    if h <= 2048 and w <= 2048:
        glctx = dr.RasterizeCudaContext()
    else:
        glctx = dr.RasterizeGLContext()

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
    rgbs, _ = dr.interpolate(vertex_colors.unsqueeze(0), rast, f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(vertex_colors[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

    # masked query 
    rgbs = rgbs.view(-1, 3)
    mask = (mask > 0).view(-1)

    feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

    if mask.any():
        feats[mask] = rgbs[mask].float()

    feats = feats.view(h, w, -1)
    mask = mask.view(h, w)

    # quantize [0.0, 1.0] to [0, 255]
    feats = (feats.cpu().numpy() * 255).astype(np.uint8)

    ### NN search as an antialiasing ...
    mask = mask.cpu().numpy()

    inpaint_region = binary_dilation(mask, iterations=3)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=2)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

    albedo = feats
    mask = (mask * 255).astype(np.uint8)[..., np.newaxis]

    if save_path is not None:
        image = np.concatenate((albedo, mask), axis=-1)
        Image.fromarray(image).save(save_path)
    
    if to_torch:
        albedo = torch.tensor(albedo / 255.0, dtype=torch.float, device=device)
        mask = torch.tensor(mask / 255.0, dtype=torch.float, device=device)

    return albedo, mask


def convert_vertex_indices_to_face_indices(vertex_indices: list, faces: np.ndarray):
    """
    Args:
        vertex_indices: list, [N]
        faces: np.ndarray, [F, 3]
    Return:
        face_indices: list, [M]
    """
    face_indices = []
    vertex_set = set(vertex_indices)
    for fid, face in enumerate(faces):
        cnt = 0
        for vid in face:
            if vid.item() in vertex_set:
                cnt += 1
        if cnt == 3:
            face_indices.append(fid)
    return face_indices
