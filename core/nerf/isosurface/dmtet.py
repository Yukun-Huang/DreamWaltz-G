import torch
from loguru import logger
import numpy as np
import os.path as osp


TETS_DIR = 'assets/tets'


def load_tets(resolution):
    tets_path = osp.join(TETS_DIR, '{}_tets.npz'.format(resolution))
    logger.info(f"Loading tets from {tets_path}")
    tets = np.load(tets_path)
    return tets


class DMTet():
    def __init__(self, device):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [ 1,  0,  2, -1, -1, -1],
            [ 4,  0,  3, -1, -1, -1],
            [ 1,  4,  2,  1,  3,  4],
            [ 3,  1,  5, -1, -1, -1],
            [ 2,  3,  0,  2,  5,  3],
            [ 1,  4,  0,  1,  5,  4],
            [ 4,  2,  5, -1, -1, -1],
            [ 4,  5,  2, -1, -1, -1],
            [ 4,  1,  0,  4,  5,  1],
            [ 3,  2,  0,  3,  5,  2],
            [ 1,  3,  5, -1, -1, -1],
            [ 4,  1,  2,  4,  3,  1],
            [ 3,  0,  4, -1, -1, -1],
            [ 2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=device)
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device=device)
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device=device)
    
    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # pos_nx3: verts + deform, [N, 3]
        # sdf_n:   sdf, [N]
        # tet_fx4: indices, [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0  # [N]
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1) # [F,]
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device=self.device)
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        return verts, faces

