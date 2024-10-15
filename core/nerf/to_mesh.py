import numpy as np
import torch
import mcubes
import cv2
import os
import pymeshlab as pml

from .nerf_utils import custom_meshgrid
from .to_point_cloud import latent_to_rgb


def poisson_mesh_reconstruction(points, normals=None):
    # points/normals: [N, 3] np.ndarray

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    # visualize
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize
    o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f'[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}')

    return vertices, triangles


def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=True):
    # optimal placement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh') # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f'[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True, remesh_size=0.01):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices() # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(threshold=pml.Percentage(v_pct)) # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces() # faces defined by the same verts
    ms.meshing_remove_null_faces() # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(min_d))
    
    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(remesh_size))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def _export(self, v, f, path, h0=2048, w0=2048, ssaa=1, name=''):
    # v, f: torch Tensor
    device = v.device
    v_np = v.cpu().numpy() # [N, 3]
    f_np = f.cpu().numpy() # [M, 3]

    print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

    # unwrap uvs
    import xatlas
    import nvdiffrast.torch as dr
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 4 # for faster unwrap...
    atlas.generate(chart_options=chart_options)
    vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

    # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

    # render uv maps
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    if ssaa > 1:
        h = int(h0 * ssaa)
        w = int(w0 * ssaa)
    else:
        h, w = h0, w0
    
    if h <= 2048 and w <= 2048:
        self.glctx = dr.RasterizeCudaContext()
    else:
        self.glctx = dr.RasterizeGLContext()

    rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

    # masked query 
    xyzs = xyzs.view(-1, 3)
    mask = (mask > 0).view(-1)

    feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

    if mask.any():
        xyzs = xyzs[mask] # [M, 3]

        # batched inference to avoid OOM
        all_feats = []
        head = 0
        while head < xyzs.shape[0]:
            tail = min(head + 640000, xyzs.shape[0])
            results_ = self.density(xyzs[head:tail])

            rgbs = latent_to_rgb(results_['albedo'].float())  # [N, 4] -> [N, 3]

            all_feats.append(rgbs)
            head += 640000

        feats[mask] = torch.cat(all_feats, dim=0)

    feats = feats.view(h, w, -1)
    mask = mask.view(h, w)

    # quantize [0.0, 1.0] to [0, 255]
    feats = feats.cpu().numpy()
    feats = (feats * 255).astype(np.uint8)

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

    feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

    # do ssaa after the NN search, in numpy
    if ssaa > 1:
        feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

    os.makedirs(path, exist_ok=True)

    cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

    # save obj (v, vt, f /)
    obj_file = os.path.join(path, f'{name}mesh.obj')
    mtl_file = os.path.join(path, f'{name}mesh.mtl')

    print(f'[INFO] writing obj mesh to {obj_file}')
    with open(obj_file, "w") as fp:
        fp.write(f'mtllib {name}mesh.mtl \n')
        
        print(f'[INFO] writing vertices {v_np.shape}')
        for v in v_np:
            fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
    
        print(f'[INFO] writing vertices texture coords {vt_np.shape}')
        for v in vt_np:
            fp.write(f'vt {v[0]} {1 - v[1]} \n') 

        print(f'[INFO] writing faces {f_np.shape}')
        fp.write(f'usemtl mat0 \n')
        for i in range(len(f_np)):
            fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

    with open(mtl_file, "w") as fp:
        fp.write(f'newmtl mat0 \n')
        fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
        fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
        fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
        fp.write(f'Tr 1.000000 \n')
        fp.write(f'illum 1 \n')
        fp.write(f'Ns 0.000000 \n')
        fp.write(f'map_Kd {name}albedo.png \n')


@torch.inference_mode()
def export_mesh(self, path, resolution=None, density_thresh=None, decimate_target=-1, split_size=128):

    os.makedirs(path, exist_ok=True)

    if self.dmtet:

        sdf = self.sdf
        deform = torch.tanh(self.deform) / self.opt.tet_grid_size

        vertices, triangles = self.dmtet_model(self.verts + deform, sdf, self.indices)

        vertices = vertices.detach().cpu().numpy()
        triangles = triangles.detach().cpu().numpy()

    else:
        if resolution is None:
            resolution = self.grid_size

        if density_thresh is None:
            if self.cuda_ray:
                density_thresh = min(self.mean_density, self.density_thresh) \
                    if np.greater(self.mean_density, 0) else self.density_thresh
            else:
                density_thresh = self.density_thresh

        # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
        if self.density_activation == 'softplus':
            density_thresh = density_thresh * 25

        sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

        # query
        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)
        S = split_size

        print(f'[INFO] start sigma query with resolution {resolution}...')

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = self.density(pts.to(self.aabb_train.device))
                    sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = \
                        val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

        print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

        vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        print(f'[INFO] marching cubes done!')

    # clean
    vertices = vertices.astype(np.float32)
    triangles = triangles.astype(np.int32)
    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)

    print(f'[INFO] mesh cleaned')

    # decimation
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

    print(f'[INFO] mesh decimated')

    v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
    f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

    # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    # mesh.export(os.path.join(path, f'mesh.ply'))
    # print(f'[INFO] exported ply file to {path}')

    _export(self, v, f, path)
