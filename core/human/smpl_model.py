import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from random import choice, random
from typing import Iterable, Optional, List, Union, Tuple
import json
import smplx
from smplx.lbs import batch_rodrigues
from smplx.body_models import SMPLX, SMPLXOutput
from configs.paths import HUMAN_TEMPLATES as MODEL_ROOT
from .smpl_utils import build_human_body_prior, smpl_to_openpose


class SMPLModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        device: torch.device,
        batch_size: int = 1,
        use_layer: bool = False,
        use_vposer: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert model_type in ('smpl', 'smplh', 'smplx')

        self.device = device
        self.use_layer = use_layer
        self.model_type = model_type
        self.batch_size = batch_size

        self.model = self.build_smpl_model(**kwargs)
        self.num_vertices = self.model.get_num_verts()
        self.num_triangles = self.model.get_num_faces()
        self.connected_vertices = self.get_connected_vertices()

        if use_vposer:
            self.vposer = build_human_body_prior()
            self.vposer.requires_grad_(False)
        else:
            self.vposer = None

        self.register_buffer('triangles', torch.tensor(self.model.faces.astype(np.int64)))
        self.register_buffer('pelvis_position', self.get_pelvis_position())
        self.to(device)

    def build_smpl_model(
        self,
        gender: str = 'neutral',
        age: str = 'adult',
        # for body
        # num_betas: int = 10,
        num_betas: int = 300,
        # for hand
        num_pca_comps: int = 12,
        use_pca: bool = False,
        flat_hand_mean: bool = False,
        # for face
        num_expression_coeffs: int = 100,
        use_face_contour: bool = True,
        use_smplx_2020_neutral: bool = False,
        **kwargs,
    ) -> SMPLX:

        model_type = self.model_type

        def joint_mapper(joints, vertices=None):
            keypoints_idx = smpl_to_openpose(model_type, openpose_format='coco18')
            keypoints = joints[..., keypoints_idx, :]
            return keypoints
        
        # SMPL Configs
        if use_smplx_2020_neutral:
            assert model_type == 'smplx' and gender == 'neutral', \
                f'Only support SMPLX_NEUTRAL_2020 for smplx and neutral, but got {model_type} and {gender}.'
            model_path = osp.join(MODEL_ROOT, 'smplx', 'SMPLX_NEUTRAL_2020.npz')
        else:
            model_path = MODEL_ROOT

        smpl_cfgs = {
            'model_path': model_path,
            'model_type': model_type,
            'ext': 'npz',
            'num_betas': num_betas,
            'joint_mapper': joint_mapper,
            'gender': gender,
            'age': age,
        }

        if model_type in ('smplh', 'smplx'):
            smpl_cfgs['use_pca'] = use_pca
            smpl_cfgs['flat_hand_mean'] = flat_hand_mean
        
        if model_type == 'smplx':
            smpl_cfgs['use_face_contour'] = use_face_contour
            smpl_cfgs['num_expression_coeffs'] = num_expression_coeffs  # 10 ~ 100
            smpl_cfgs['num_pca_comps'] = num_pca_comps  # 6 ~ 45
        
        if age == 'kid':
            smpl_cfgs['kid_template_path'] = \
                osp.join(MODEL_ROOT, model_type, 'smplx_kid_template.npy')
            # https://github.com/pixelite1201/agora_evaluation/blob/master/docs/kid_model.md
            # It is recommended to use gender= 'male' for male kids and gender='neutral' for female kids.
            if gender == 'female':
                smpl_cfgs['gender'] = 'neutral'

        if self.use_layer:
            model = smplx.build_layer(**smpl_cfgs, **kwargs)
        else:
            smpl_cfgs['batch_size'] = self.batch_size
            model = smplx.create(**smpl_cfgs, **kwargs)
        
        if model_type in ('smplh', 'smplx') and not model.use_pca:
            model.register_buffer('left_hand_components', torch.tensor(model.np_left_hand_components, dtype=model.dtype))
            model.register_buffer('right_hand_components', torch.tensor(model.np_right_hand_components, dtype=model.dtype))

        if age == 'kid':
            kid_interpolation_rate = 0.7
            model.betas.data[:, -1] = kid_interpolation_rate
        
        return model

    @torch.no_grad()
    def get_pelvis_position(self) -> torch.Tensor:
        joint_mapper = self.model.joint_mapper
        self.model.joint_mapper = None
        smpl_outputs: SMPLXOutput = self.model.forward()
        self.model.joint_mapper = joint_mapper
        pelvis_position = smpl_outputs.joints[:1, 0, :]
        # pelvis_position[:, 0] = 0.
        # pelvis_position[:, 1] -= 0.05
        # pelvis_position[:, 2] = 0.
        return pelvis_position

    @torch.no_grad()
    def sample_body_pose(self, batch_size) -> torch.Tensor:
        body_pose = self.vposer.sample_poses(num_poses=batch_size)['pose_body']  # tensor with shape of (N, 21, 3)
        if self.model_type == 'smpl':
            body_pose_hands = torch.zeros((batch_size, 2, 3), device=body_pose.device)
            body_pose = torch.cat((body_pose, body_pose_hands), dim=1)
        body_pose = body_pose.contiguous().view(batch_size, -1)  # body_pose shape = (N, 63)
        return body_pose

    @torch.no_grad()
    def sample_face_experssion(self, batch_size) -> torch.Tensor:
        return torch.randn((batch_size, self.model.num_expression_coeffs), device=self.device)

    @torch.no_grad()
    def sample_hand_pose(self, batch_size) -> Tuple[torch.Tensor]:

        assert self.model.flat_hand_mean == False

        if self.model.use_pca:
            left_hand_pose = torch.randn((batch_size, self.model.num_pca_comps), device=self.device)
            right_hand_pose = torch.randn((batch_size, self.model.num_pca_comps), device=self.device)
        else:
            left_hand_pose = torch.randn((batch_size, self.model.num_pca_comps), device=self.device)
            right_hand_pose = torch.randn((batch_size, self.model.num_pca_comps), device=self.device)
            
            left_hand_pose = torch.einsum('bi,ij->bj', [left_hand_pose, self.model.left_hand_components])
            right_hand_pose = torch.einsum('bi,ij->bj', [right_hand_pose, self.model.right_hand_components])

        return left_hand_pose, right_hand_pose

    def get_canonical_pose(self, pose_type: str, batch_size: int = 1) -> dict:
        device = self.device
        num_joints = 23 if self.model_type == 'smpl' else 21

        body_pose = torch.zeros((batch_size, num_joints, 3), device=device)
        if pose_type == 'canonical-T':
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # right_hip
        elif pose_type == 'canonical-T-adjust':  # Adjust hip posture to avoid foot adhesion
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +np.pi/30], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -np.pi/30], device=device)  # right_hip
        elif pose_type == 'canonical-Y':
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # left_shoulder
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # right_shoulder
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # right_hip
        elif pose_type == 'canonical-Y-adjust':  # Adjust hip posture to avoid foot adhesion
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # left_shoulder
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # right_shoulder
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +np.pi/30], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -np.pi/30], device=device)  # right_hip
        elif pose_type == 'canonical-A':
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # left_shoulder
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # right_shoulder
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # right_hip
        elif pose_type == 'canonical-A-adjust':  # Adjust hip posture to avoid foot adhesion
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi/4], device=device)  # left_shoulder
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +np.pi/4], device=device)  # right_shoulder
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +np.pi/30], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -np.pi/30], device=device)  # right_hip
        elif pose_type == 'canonical-R':
            shoulder_angle = np.random.uniform(-np.pi/4, np.pi/4)
            hip_angle = np.random.uniform(np.pi/30, np.pi/4)
            body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -shoulder_angle], device=device)  # left_shoulder
            body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +shoulder_angle], device=device)  # right_shoulder
            body_pose[:, 0, :] = torch.tensor([0.0, 0.0, +hip_angle], device=device)  # left_hip
            body_pose[:, 1, :] = torch.tensor([0.0, 0.0, -hip_angle], device=device)  # right_hip
        else:
            assert 0, pose_type
        
        return body_pose.reshape(batch_size, -1)

    def get_smpl_inputs(
        self,
        pose_type: str,
        batch_size: int = 1,
        flat_hand: bool = True,
        centralize_pelvis: bool = True,
        canonical_mixup_prob: float = 0.5,
        **kwargs,
    ) -> dict:
        """
        - Reference:
          from smplx.joint_names import SMPL_JOINT_NAMES, SMPLH_JOINT_NAMES, JOINT_NAMES
        """
        # Initialize SMPL Parameters
        smpl_params = {}
        device = self.device

        if pose_type.startswith('random') and random() < canonical_mixup_prob:
            pose_type = 'canonical-R'

        # Pose
        if pose_type == 'vposer':
            smpl_params['body_pose'] = self.sample_body_pose(batch_size=batch_size)

        elif pose_type.startswith('random'):
            if '-' in pose_type:
                # e.g., "random-body,hand,expr"
                parts = pose_type.split('-')[-1].split(',')
            else:
                parts = ['body', 'hand', 'expr']
            
            for part in parts:
                if part == 'expr':
                    smpl_params['expression'] = self.sample_face_experssion(batch_size=batch_size)
                elif part == 'hand':
                    smpl_params['left_hand_pose'], smpl_params['right_hand_pose'] = self.sample_hand_pose(batch_size=batch_size)
                elif part == 'body':
                    smpl_params['body_pose'] = self.sample_body_pose(batch_size=batch_size)
                else:
                    assert 0, part

        elif pose_type.startswith('canonical'):

            if pose_type == 'canonical-choice':
                rand_pose_type = choice([
                    'canonical-Y', 'canonical-T', 'canonical-A',
                    'canonical-Y-adjust', 'canonical-T-adjust', 'canonical-A-adjust',
                ])
                smpl_params['body_pose'] = self.get_canonical_pose(rand_pose_type, batch_size=batch_size)

            elif pose_type == 'canonical-loop':
                body_pose_A = self.get_canonical_pose('canonical-A-adjust', batch_size=batch_size)
                body_pose_B = self.get_canonical_pose('canonical-Y', batch_size=batch_size)
                training_ratio = kwargs.pop('training_ratio')
                smpl_params['body_pose'] = body_pose_A * (1 - training_ratio) + body_pose_B * training_ratio
            
            elif pose_type == 'canonical-loop2':
                body_pose_A = self.get_canonical_pose('canonical-A-adjust', batch_size=batch_size)
                body_pose_B = self.get_canonical_pose('canonical-Y', batch_size=batch_size)
                training_ratio = kwargs.pop('training_ratio') * 2  # [0.0, 2.0]
                if training_ratio > 1.0:
                    training_ratio = 2.0 - training_ratio
                smpl_params['body_pose'] = body_pose_A * (1 - training_ratio) + body_pose_B * training_ratio
            
            else:
                smpl_params['body_pose'] = self.get_canonical_pose(pose_type, batch_size=batch_size)
        
            if flat_hand and self.model_type in ('smplh', 'smplx'):
                if not self.model.flat_hand_mean:
                    assert self.model.use_pca == False
                    smpl_params['left_hand_pose'] = - self.model.left_hand_mean.to(device)
                    smpl_params['right_hand_pose'] = - self.model.right_hand_mean.to(device)
        
        else:
            assert 0, pose_type
        
        # Translation
        if centralize_pelvis:
            smpl_params['transl'] = - self.pelvis_position.to(device)

        # Update
        for k, v in kwargs.items():
            if v is not None:
                assert k not in smpl_params.keys()
                smpl_params[k] = v

        # Return
        return smpl_params

    def get_connected_vertices(self) -> list:
        connected_vertices = [[] for _ in range(self.num_vertices)]

        for vids in self.model.faces:
            i1, i2, i3 = vids.tolist()
            connected_vertices[i1].extend([i2, i3])
            connected_vertices[i2].extend([i1, i3])
            connected_vertices[i3].extend([i1, i2])

        for i in range(len(connected_vertices)):
            connected_vertices[i] = sorted(list(set(connected_vertices[i])))  # len = 3 ~ 32
        
        return connected_vertices

    def to_rot_matrix(self, poses) -> torch.Tensor:
        if poses.ndim == 2:
            batch_size = poses.shape[0]
            return batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])
        else:
            assert poses.size(-1) == poses.size(-2) == 3
            return poses

    def to_open3d_mesh(self, smpl_outputs: SMPLXOutput):
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(smpl_outputs.vertices[0].detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.model.faces)
        return mesh

    @torch.no_grad()
    def forward(
        self,
        body_pose:Optional[torch.Tensor]=None,
        global_orient:Optional[torch.Tensor]=None,
        **kwargs,
    ) -> SMPLXOutput:
        """
        Input:
            body_pose: torch.Tensor, shape = (N, V*3) or (N, V, 3, 3)
            global_orient: torch.Tensor, shape = (N, 3) or (N, 3, 3)
            betas: torch.Tensor, shape = (N, 10)
            transl: torch.Tensor, shape = (N, 3)
        """
        # Axis-aligned Vector to Rotation Matrix
        if self.use_layer:
            if torch.is_tensor(body_pose):
                body_pose = self.to_rot_matrix(body_pose)
            if torch.is_tensor(global_orient):
                global_orient = self.to_rot_matrix(global_orient)
        # Inference
        return self.model.forward(body_pose=body_pose, global_orient=global_orient, return_verts=True, **kwargs)


class SMPLSemantics:
    def __init__(self, smpl_model: SMPLModel) -> None:
        super().__init__()

        # basic params
        self.faces = smpl_model.model.faces
        self.model_type = smpl_model.model_type
        self.num_vertices = smpl_model.num_vertices
        self.connected_vertices = smpl_model.connected_vertices

        # load labels
        label_to_vertices_RAW = self.load_segmentation_labels(self.model_type)
        label_to_vertices_FLAME = self.load_FLAME_segmentation_labels(self.model_type)
        label_to_vertices_MANO = self.load_MANO_segmentation_labels(self.model_type)
        for k, v in label_to_vertices_FLAME.items():
            label_to_vertices_RAW[k + '_FLAME'] = v
        for k, v in label_to_vertices_MANO.items():
            label_to_vertices_RAW[k + '_MANO'] = v

        # convert to new segmentation
        self.label_to_vertices, self.label_to_faces = self.convert_to_new_segmentation(label_to_vertices_RAW)
        self.labels = sorted(list(self.label_to_vertices.keys()))

    @staticmethod
    def load_segmentation_labels(model_type):
        """
            smpl_labels = [
                'rightShoulder', 'rightArm', 'rightForeArm', 'rightHand', 'rightHandIndex1',
                'leftShoulder', 'leftArm', 'leftForeArm', 'leftHand', 'leftHandIndex1',
                'rightLeg', 'rightUpLeg', 'rightFoot', 'rightToeBase',
                'leftLeg', 'leftUpLeg', 'leftFoot', 'leftToeBase',
                'spine', 'spine1', 'spine2', 'hips', 'head', 'neck',
            ]

            smplx_labels = smpl_labels + ['leftEye', 'rightEye', 'eyeballs']
        """
        json_path = osp.join(MODEL_ROOT, model_type, f'{model_type}_vert_segmentation.json')
        with open(json_path, 'r') as fio:
            label_to_vertices = json.load(fio)
        return label_to_vertices

    @staticmethod
    def load_FLAME_segmentation_labels(model_type):
        """
            FLAME_labels = [
                'eye_region', 'left_eyeball', 'right_eyeball', 'right_eye_region', 'left_eye_region',
                'neck', 'boundary', 'face', 'forehead', 'scalp',
                'left_ear', 'right_ear', 'lips', 'nose',
            ]
        """
        if model_type != 'smplx':
            return {}
        vids_FLAME = np.load(osp.join(MODEL_ROOT, model_type, 'FLAME_vertex_ids.npy'))
        segs_FLAME = np.load(osp.join(MODEL_ROOT, 'flame', 'FLAME_masks.pkl'), allow_pickle=True, encoding='latin1')
        label_to_vertices = {}
        for k, v in segs_FLAME.items():
            label_to_vertices[k] = list(vids_FLAME[v])
        return label_to_vertices

    @staticmethod
    def load_MANO_segmentation_labels(model_type):
        if model_type != 'smplx':
            return {}
        vertex_ids_MANO = np.load(osp.join(MODEL_ROOT, model_type, 'MANO_vertex_ids.pkl'), allow_pickle=True)
        return {
            'left_hand': vertex_ids_MANO['left_hand'].tolist(),
            'right_hand': vertex_ids_MANO['right_hand'].tolist(),
        }

    @staticmethod
    def convert_vertex_indices_to_face_indices(vertex_indices: list, faces: np.ndarray, strict: bool = True):
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
            if strict and cnt == 3:
                face_indices.append(fid)
            elif not strict and cnt > 0:
                face_indices.append(fid)
        return sorted(list(set(face_indices)))

    def convert_to_new_segmentation(self, label_to_vertices):
        # Get Label-to-Vertex Mapping
        res = {}
        
        res['scalp'] = label_to_vertices['scalp_FLAME']
        res['face'] = label_to_vertices['face_FLAME']
        res['eye region'] = label_to_vertices['eye_region_FLAME']
        res['eyes'] = label_to_vertices['eyeballs']
        res['neck'] = label_to_vertices['neck']
        res['spine'] = label_to_vertices['spine'] + label_to_vertices['spine1'] + label_to_vertices['spine2']
        res['shoulders'] = label_to_vertices['leftShoulder'] + label_to_vertices['rightShoulder']
        res['torso'] = label_to_vertices['spine'] + label_to_vertices['spine1'] + label_to_vertices['spine2'] + \
            label_to_vertices['leftShoulder'] + label_to_vertices['rightShoulder']
        
        res['hand_left'] = label_to_vertices['leftHand'] + label_to_vertices['leftHandIndex1']
        res['hand_right'] = label_to_vertices['rightHand'] + label_to_vertices['rightHandIndex1']

        res['hand_left_index1'] = label_to_vertices['leftHandIndex1']
        res['hand_right_index1'] = label_to_vertices['rightHandIndex1']

        res['hands'] = label_to_vertices['leftHand'] + label_to_vertices['leftHandIndex1'] + \
            label_to_vertices['rightHand'] + label_to_vertices['rightHandIndex1']
        
        res['upper arms'] = label_to_vertices['leftArm'] + label_to_vertices['rightArm']

        res['forearms'] = label_to_vertices['leftForeArm'] + label_to_vertices['rightForeArm']
        res['forearm_left'] = label_to_vertices['leftForeArm']
        res['forearm_right'] = label_to_vertices['rightForeArm']
        
        res['hips'] = label_to_vertices['hips']
        res['lower legs'] = label_to_vertices['leftLeg'] + label_to_vertices['rightLeg']
        res['upper legs'] = label_to_vertices['leftUpLeg'] + label_to_vertices['rightUpLeg']
        
        res['feet'] = label_to_vertices['leftFoot'] + label_to_vertices['leftToeBase'] + \
            label_to_vertices['rightFoot'] + label_to_vertices['rightToeBase']
        
        res['skin'] = set([i for i in range(self.num_vertices)])
        res['skin'].difference_update(set(label_to_vertices['eyeballs']))

        # Get Extra Label-to-Vertex Mapping
        res['wrist_left'] = set.intersection(set(res['forearm_left']), set(res['hand_left']))
        res['wrist_right'] = set.intersection(set(res['forearm_right']), set(res['hand_right']))
        for _ in range(3):
            extra_vertices = []
            for i in res['wrist_left']:
                extra_vertices.extend(self.connected_vertices[i])
            res['wrist_left'] = res['wrist_left'].union(set(extra_vertices))
            extra_vertices = []
            for i in res['wrist_right']:
                extra_vertices.extend(self.connected_vertices[i])
            res['wrist_right'] = res['wrist_right'].union(set(extra_vertices))
        
        res['wrists'] = set.union(res['wrist_left'], res['wrist_right'])

        # Get Label-to-Face Mapping
        label_to_vertices = {}
        label_to_faces = {}

        for k in res.keys():
            vertex_indices = sorted(list(set(res[k])))
            label_to_vertices[k] = vertex_indices
            if k in ['wrist_left', 'wrist_right']:
                label_to_faces[k] = self.convert_vertex_indices_to_face_indices(vertex_indices, self.faces, strict=False)
            else:
                label_to_faces[k] = self.convert_vertex_indices_to_face_indices(vertex_indices, self.faces, strict=True)

        return label_to_vertices, label_to_faces

    def __call__(self, select_parts: Union[str, List[str]]) -> List[int]:
        if isinstance(select_parts, str):
            vertex_indices = self.label_to_vertices[select_parts]
            face_indices = self.label_to_faces[select_parts]
        else:
            vertex_indices, face_indices = set(), set()
            for p in select_parts:
                vertex_indices = vertex_indices.union(self.label_to_vertices[p])
                face_indices = face_indices.union(self.label_to_faces[p])
            vertex_indices = sorted(list(vertex_indices))
            face_indices = sorted(list(face_indices))
        return vertex_indices, face_indices

    def visualize(self, vertices, faces, select_parts=None, alpha=1.0):
        import trimesh
        from matplotlib import colormaps as mpl_cm, colors as mpl_colors

        n_vertices = vertices.shape[0]

        vertex_labels = np.zeros(n_vertices)

        if select_parts is not None:
            vids, fids = self(select_parts)
            vertex_labels[vids] = 1
        else:
            for part_idx, (k, v) in enumerate(self.label_to_vertices.items(), start=1):
                vertex_labels[v] = part_idx

        cm = mpl_cm.get_cmap('jet')
        norm_gt = mpl_colors.Normalize()

        vertex_colors = np.ones((n_vertices, 4))
        vertex_colors[:, 3] = alpha
        vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

        mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertex_colors)
        # mesh.show(background=(0,0,0,0))
        return mesh


class SemanticSMPLModel(SMPLModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.model_type == 'smplx':
            self.semantics =  SMPLSemantics(self)
        else:
            self.semantics = None

    def get_semantic_indices(self, select_parts: Union[str, List[str]]) -> List[int]:
        return self.semantics(select_parts)
