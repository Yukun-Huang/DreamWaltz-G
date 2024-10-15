import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
# import igl
from typing import Union
from smplx import SMPL, SMPLX
from smplx.lbs import blend_shapes, batch_rodrigues, vertices2joints, batch_rigid_transform
from copy import deepcopy
from typing import NewType, Union, Optional, Tuple
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply


class RigidTransform:
    SE3: Tensor
    R: Tensor
    T: Tensor

    def __init__(self, SE3=None, R=None, T=None) -> None:
        if SE3 is not None:
            SE3 = SE3
        elif R is not None and T is not None:
            SE3 = self.RT_to_SE3(R, T)
        elif R is not None:
            SE3 = self.R_to_SE3(R)
        elif T is not None:
            SE3 = self.T_to_SE3(T)
        else:
            raise NotImplementedError
        R, T = self.SE3_to_RT(SE3)
        self.SE3 = SE3
        self.R = R
        self.T = T

    @property
    def shape(self):
        return self.SE3.shape[:-2]

    @staticmethod
    def R_to_SE3(R: Tensor):
        """
            Parameters
            ----------
            R : torch.tensor [..., 3, 3]

            Returns
            -------
            SE3 : torch.Tensor [..., 4, 4]
        """
        SE3 = torch.eye(4, 4, dtype=R.dtype, device=R.device)
        SE3 = SE3.expand(*R.shape[:-2], 4, 4).contiguous()
        SE3[..., :3, :3] = R
        return SE3

    @staticmethod
    def T_to_SE3(T: Tensor):
        """
            Parameters
            ----------
            T : torch.tensor [..., 3]

            Returns
            -------
            SE3 : torch.Tensor [..., 4, 4]
        """
        SE3 = torch.eye(4, 4, dtype=T.dtype, device=T.device)
        SE3 = SE3.expand(*T.shape[:-1], 4, 4).contiguous()
        SE3[..., :3, 3] = T
        return SE3

    @staticmethod
    def RT_to_SE3(R: Tensor, T: Tensor):
        """
            Parameters
            ----------
            R : torch.tensor [..., 3, 3]
            T : torch.tensor [..., 3]

            Returns
            -------
            SE3 : torch.Tensor [..., 4, 4]
        """
        SE3 = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
        SE3[..., :3, :3] = R
        SE3[..., :3, 3] = T
        SE3[..., 3, 3] = 1.0
        return SE3

    @staticmethod
    def SE3_to_RT(SE3: Tensor):
        """
            Parameters
            ----------
            SE3 : torch.Tensor [..., 4, 4]

            Returns
            -------
            R : torch.tensor [..., 3, 3]
            T : torch.tensor [..., 3]
        """
        R = SE3[..., :3, :3]
        T = SE3[..., :3, 3]
        return R, T

    @staticmethod
    def _invert_transformation(SE3: Tensor):
        """
            Parameters
            ----------
            SE3 : torch.Tensor [..., 4, 4]

            Returns
            -------
            SE3 : torch.Tensor [..., 4, 4]
        """
        # Ensure the input matrices have shape [..., 4, 4]
        assert SE3.shape[-2:] == (4, 4), "Input matrices must have shape [..., 4, 4]"

        # Ensure the last row of the matrix is [0, 0, 0, 1]
        eye = torch.tensor([0, 0, 0, 1], dtype=SE3.dtype, device=SE3.device)
        SE3[..., 3, :] = eye

        # Compute the inverse of the rotation part (transpose)
        rotation_transpose = SE3[..., :3, :3].transpose(-1, -2)

        # Compute the inverse of the translation part
        translation = SE3[..., :3, 3]
        translation_inv = -torch.matmul(rotation_transpose, translation.unsqueeze(-1)).squeeze(-1)

        # Assemble the inverse matrix
        inverse_matrices = torch.zeros_like(SE3)
        inverse_matrices[..., :3, :3] = rotation_transpose
        inverse_matrices[..., :3, 3] = translation_inv
        inverse_matrices[..., 3, 3] = 1.0

        return inverse_matrices
    
    def inverse(self):
        # dtype = self.SE3.dtype
        # SE3_inv = torch.inverse(self.SE3.float()).to(dtype=dtype)
        SE3_inv = self._invert_transformation(self.SE3)
        return RigidTransform(SE3=SE3_inv)

    def compose(self, *others):
        """
        Args:
            *others: Any number of RigidTransform objects

        Returns:
            A new RigidTransform
        """
        SE3 = self.SE3.clone()
        for other in others:
            if not isinstance(other, RigidTransform):
                msg = "Only possible to compose RigidTransform objects; got %s"
                raise ValueError(msg % type(other))
            SE3 = other.SE3 @ SE3
        return RigidTransform(SE3=SE3)

    @staticmethod
    def correct_rotation_matrices(R):
        # Perform QR decomposition on R
        Q, R_ = torch.linalg.qr(R)
        
        # Ensure the determinant of Q is positive to handle reflections
        Q = Q * torch.sign(torch.det(Q)).unsqueeze(-1).unsqueeze(-1)
        
        return Q
    
    def index(self, indices:Tensor):
        return RigidTransform(SE3=self.SE3[indices])
    
    def weight(self, weights:Tensor, qr_correct:bool=False):
        if qr_correct: # Too slow!
            R = torch.einsum('nj,jkl->nkl', weights, self.R)
            T = torch.einsum('nj,jk->nk', weights, self.T)
            return RigidTransform(R=self.correct_rotation_matrices(R), T=T)
        else:
            return RigidTransform(SE3=torch.einsum('nj,jkl->nkl', weights, self.SE3))

    @staticmethod
    def _transform_points(pts:Tensor, R:Tensor, T:Tensor):
        return torch.matmul(R, pts.unsqueeze(-1))[..., :, 0] + T

    @staticmethod
    def _inverse_transform_points(pts:Tensor, R:Tensor, T:Tensor):
        return torch.matmul(torch.inverse(R), (pts - T).unsqueeze(-1))[..., :, 0]

    def transform_points(self, points:Tensor, indices:Tensor=None, weights:Tensor=None):
        """
        Args:
            points: torch.Tensor [N, 3]
            indices: torch.Tensor [N]
            weights: torch.Tensor [N, J]
        Vars:
            R: torch.Tensor [V or J, 3, 3]
            T: torch.Tensor [V or J, 3]
        
        Returns:
            torch.Tensor [N, 3]
        """
        assert indices is None or weights is None
        R, T = self.R, self.T
        if indices is not None:
            R, T = R[indices], T[indices]
        if weights is not None:
            R = torch.einsum('nj,jkl->nkl', weights, R)
            T = torch.einsum('nj,jk->nk', weights, T)
        return self._transform_points(points, R, T)

    def transform_quaternions(
        self,
        quaternions: Tensor,
        indices: Tensor = None,
        weights: Tensor = None,
        rotation_mode: str = 'quaternion',
        flip_rotation_axis: bool = False,
    ):
        """
        Args:
            quaternions: torch.Tensor [N, 4]
            indices: torch.Tensor [N]
            weights: torch.Tensor [N, J]
        Vars:
            R: torch.Tensor [V, 3, 3]

        Returns:
            torch.Tensor [..., 4]
        """
        assert indices is None or weights is None
        if indices is not None:
            R = self.R[indices]
        if weights is not None:
            R = torch.einsum('nj,jkl->nkl', weights, self.R)
        
        if flip_rotation_axis:  # Fix the direction of the axis
            rotations = quaternion_to_matrix(quaternions)
            rotations[:, [1, 2], :] *= -1
            rotations = R @ rotations
            rotations[:, [1, 2], :] *= -1
            return matrix_to_quaternion(rotations)
        
        if rotation_mode == 'matrix':
            rotations = quaternion_to_matrix(quaternions)  # [N, 4] -> [N, 3, 3]
            return matrix_to_quaternion(R @ rotations)
        elif rotation_mode == 'quaternion':
            quaternions_of_R = matrix_to_quaternion(R)  # [N, 3, 3] -> [N, 4]
            return quaternion_multiply(quaternions_of_R, quaternions)
        else:
            assert 0, rotation_mode

    def __repr__(self) -> str:
        return f"SE3: {self.SE3},\r\nR: {self.R},\r\nT: {self.T}"

    def squeeze(self, dim=0):
        self.SE3 = self.SE3.squeeze(dim=dim)
        self.R = self.R.squeeze(dim=dim)
        self.T = self.T.squeeze(dim=dim)
        return self


class LinearBlendSkinning(nn.Module):
    r"""Compute transforms between the observed pose and canonical pose."""

    def __init__(
        self,
        model: SMPL | SMPLX,
        learn_v_template: bool = False,
        learn_shapedirs: bool = False,
        learn_posedirs: bool = False,
        learn_J_regressor: bool = False,
        learn_lbs_weights: bool = False,
        learn_expr_dirs: bool = False,
    ):
        super().__init__()
        # smpl params
        self.num_joints = model.NUM_JOINTS + 1
        self.NUM_BODY_JOINTS = model.NUM_BODY_JOINTS
        self.faces = deepcopy(model.faces)
        self.parents = deepcopy(model.parents)
        self.betas = nn.Parameter(deepcopy(model.betas[:1, ...]), requires_grad=False)
        self.body_pose = nn.Parameter(deepcopy(model.body_pose[:1, ...]), requires_grad=False)
        self.global_orient = nn.Parameter(deepcopy(model.global_orient[:1, ...]), requires_grad=False)

        self.v_template = nn.Parameter(deepcopy(model.v_template), requires_grad=learn_v_template)
        self.shapedirs = nn.Parameter(deepcopy(model.shapedirs), requires_grad=learn_shapedirs)
        self.posedirs = nn.Parameter(deepcopy(model.posedirs), requires_grad=learn_posedirs)
        self.J_regressor = nn.Parameter(deepcopy(model.J_regressor), requires_grad=learn_J_regressor)
        self.lbs_weights = nn.Parameter(deepcopy(model.lbs_weights), requires_grad=learn_lbs_weights)
        
        # other params
        if isinstance(model, SMPLX):
            self.use_smplx = True
            self.use_pca = model.use_pca
            self.left_hand_pose = nn.Parameter(deepcopy(model.left_hand_pose[:1, ...]), requires_grad=False)
            self.right_hand_pose = nn.Parameter(deepcopy(model.right_hand_pose[:1, ...]), requires_grad=False)
            self.pose_mean = nn.Parameter(deepcopy(model.pose_mean), requires_grad=False)
            self.left_hand_components = nn.Parameter(deepcopy(model.left_hand_components), requires_grad=False)
            self.right_hand_components = nn.Parameter(deepcopy(model.right_hand_components), requires_grad=False)
            self.jaw_pose = nn.Parameter(deepcopy(model.jaw_pose[:1, ...]), requires_grad=False)
            self.leye_pose = nn.Parameter(deepcopy(model.leye_pose[:1, ...]), requires_grad=False)
            self.reye_pose = nn.Parameter(deepcopy(model.reye_pose[:1, ...]), requires_grad=False)
            self.expr_dirs = nn.Parameter(deepcopy(model.expr_dirs), requires_grad=learn_expr_dirs)
            self.expression = nn.Parameter(deepcopy(model.expression[:1, ...]), requires_grad=False)
        else:
            self.use_smplx = False

    def get_full_shape(
        self,
        betas: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ):
        betas = self.betas if betas is None else betas
        if self.use_smplx:
            expression = expression if expression is not None else self.expression
            shape_components = torch.cat([betas, expression], dim=-1)
        else:
            shape_components = betas
        if batch_size is not None:
            scale = int(batch_size / shape_components.shape[0])
            if scale > 1:
                shape_components = shape_components.expand(scale, -1)
        return shape_components

    def get_full_pose(
        self,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
    ):
        global_orient = global_orient if global_orient is not None else self.global_orient
        body_pose = body_pose if body_pose is not None else self.body_pose

        if self.use_smplx:
            left_hand_pose = left_hand_pose if left_hand_pose is not None else self.left_hand_pose
            right_hand_pose = right_hand_pose if right_hand_pose is not None else self.right_hand_pose
            jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
            leye_pose = leye_pose if leye_pose is not None else self.leye_pose
            reye_pose = reye_pose if reye_pose is not None else self.reye_pose

            if self.use_pca:
                left_hand_pose = torch.einsum('bi,ij->bj', [self.left_hand_pose, self.left_hand_components])
                right_hand_pose = torch.einsum('bi,ij->bj', [self.right_hand_pose, self.right_hand_components])

            full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
                                self.jaw_pose.reshape(-1, 1, 3),
                                self.leye_pose.reshape(-1, 1, 3),
                                self.reye_pose.reshape(-1, 1, 3),
                                left_hand_pose.reshape(-1, 15, 3),
                                right_hand_pose.reshape(-1, 15, 3)],
                                dim=1).reshape(-1, 165)

            # Add the mean pose of the model. Does not affect the body, only the
            # hands when flat_hand_mean == False
            full_pose += self.pose_mean

        else:
            full_pose = torch.cat([global_orient, body_pose], dim=1)  # [N, J*3]

        return full_pose

    def get_lbs_transform(self, betas: Tensor, pose: Tensor, pose2rot: bool = True) -> dict:
        ''' 
            Parameters
            ----------
            betas : torch.tensor BxNB
                The tensor of shape parameters
            pose : torch.tensor Bx(J + 1) * 3
                The pose parameters in axis-angle format
            pose2rot: bool, optional
                Flag on whether to convert the input pose tensor to rotation
                matrices. The default value is True. If False, then the pose tensor
                should already contain rotation matrices and have a size of
                Bx(J + 1)x9

            Returns
            -------
            dict : str -> RigidTransform
        '''
        # Get default params
        parents = self.parents
        v_template = self.v_template
        posedirs = self.posedirs
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1) if self.use_smplx else self.shapedirs
        J_regressor = self.J_regressor
        lbs_weights = self.lbs_weights

        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        shape_offsets = blend_shapes(betas, shapedirs)
        v_shaped = v_template + shape_offsets

        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
                [batch_size, -1, 3, 3])

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)

        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        transform_V_shape_offset = RigidTransform(T=shape_offsets)
        transform_V_pose_offset = RigidTransform(T=pose_offsets)

        transform_J_pose_rigid = RigidTransform(SE3=A)
        transform_V_pose_rigid = RigidTransform(SE3=T)

        return {
            "V_shape_offset": transform_V_shape_offset,
            "V_pose_offset": transform_V_pose_offset,
            "V_pose_rigid": transform_V_pose_rigid,
            "J_pose_rigid": transform_J_pose_rigid,
        }

    def forward(
        self,
        betas: Tensor = None,
        body_pose: Tensor = None,
        global_orient: Tensor = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        expression: Tensor = None,
        transl: Tensor = None,
        deform_with_shape: bool = False,
        # New
        flame_betas: Optional[Tensor] = None,
        flame_expression: Optional[Tensor] = None,
    ) -> RigidTransform:
        """
        Input:
            body_pose: torch.Tensor, shape = (N, V*3) or (N, V, 3, 3)
            global_orient: torch.Tensor, shape = (N, 3) or (N, 3, 3)
            betas: torch.Tensor, shape = (N, 10)
            transl: torch.Tensor, shape = (N, 3)
        """
        assert betas is None
        
        full_shape = self.get_full_shape(betas=betas, expression=expression,)
        full_pose = self.get_full_pose(
            body_pose=body_pose,
            global_orient=global_orient,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
        )

        transforms = self.get_lbs_transform(
            betas=full_shape,
            pose=full_pose,
            pose2rot=True,
        )
        transform_V_shape_offset = transforms['V_shape_offset']
        transform_V_pose_offset = transforms['V_pose_offset']
        transform_V_pose_rigid = transforms['V_pose_rigid']

        if deform_with_shape:
            transform_V = transform_V_shape_offset.compose(transform_V_pose_offset, transform_V_pose_rigid)
        else:
            transform_V = transform_V_pose_offset.compose(transform_V_pose_rigid)
        
        if transl is not None:
            transform_transl = RigidTransform(T=transl)
            transform_V = transform_V.compose(transform_transl)
        
        return transform_V

    def get_optimizer(self, lr: float):
        params_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                if name == 'v_template':
                    params_list.append({'params': params, 'lr': lr * 10})
                else:
                    params_list.append({'params': params, 'lr': lr})
        if len(params_list) == 0:
            return None
        else:
            return torch.optim.Adam(params_list, weight_decay=0.0)


class GeneralLinearBlendSkinning(nn.Module):
    # LBS:  3DGS -> SMPLX Vertices -> LBS Weights -> Rigid Transforms
    # GLBS: 3DGS -> LBS Weights -> Rigid Transforms

    def __init__(
        self,
        model: SMPL | SMPLX,
        learn_v_template: bool = False,
        learn_shapedirs: bool = False,
        learn_posedirs: bool = False,
        learn_J_regressor: bool = False,
        learn_lbs_weights: bool = False,
        learn_expr_dirs: bool = False,
    ):
        super().__init__()
        # smpl params
        self.num_joints = model.NUM_JOINTS + 1
        self.NUM_BODY_JOINTS = model.NUM_BODY_JOINTS
        self.faces = deepcopy(model.faces)
        self.parents = deepcopy(model.parents)
        self.betas = nn.Parameter(deepcopy(model.betas[:1, ...]), requires_grad=False)
        self.body_pose = nn.Parameter(deepcopy(model.body_pose[:1, ...]), requires_grad=False)
        self.global_orient = nn.Parameter(deepcopy(model.global_orient[:1, ...]), requires_grad=False)

        self.v_template = nn.Parameter(deepcopy(model.v_template), requires_grad=learn_v_template)
        self.shapedirs = nn.Parameter(deepcopy(model.shapedirs), requires_grad=learn_shapedirs)
        self.posedirs = nn.Parameter(deepcopy(model.posedirs), requires_grad=learn_posedirs)
        self.J_regressor = nn.Parameter(deepcopy(model.J_regressor), requires_grad=learn_J_regressor)
        self.lbs_weights = nn.Parameter(deepcopy(model.lbs_weights), requires_grad=learn_lbs_weights)

        J_template = torch.einsum('ik,ji->jk', [self.v_template.data.detach(), self.J_regressor.data.detach()])
        self.register_buffer('J_template', J_template)
        
        # other params
        if isinstance(model, SMPLX):
            self.use_smplx = True
            self.use_pca = model.use_pca
            self.left_hand_pose = nn.Parameter(deepcopy(model.left_hand_pose[:1, ...]), requires_grad=False)
            self.right_hand_pose = nn.Parameter(deepcopy(model.right_hand_pose[:1, ...]), requires_grad=False)
            self.pose_mean = nn.Parameter(deepcopy(model.pose_mean), requires_grad=False)
            self.left_hand_components = nn.Parameter(deepcopy(model.left_hand_components), requires_grad=False)
            self.right_hand_components = nn.Parameter(deepcopy(model.right_hand_components), requires_grad=False)
            self.jaw_pose = nn.Parameter(deepcopy(model.jaw_pose[:1, ...]), requires_grad=False)
            self.leye_pose = nn.Parameter(deepcopy(model.leye_pose[:1, ...]), requires_grad=False)
            self.reye_pose = nn.Parameter(deepcopy(model.reye_pose[:1, ...]), requires_grad=False)
            self.expr_dirs = nn.Parameter(deepcopy(model.expr_dirs), requires_grad=learn_expr_dirs)
            self.expression = nn.Parameter(deepcopy(model.expression[:1, ...]), requires_grad=False)
        else:
            self.use_smplx = False
        
        # buffer
        self.joints = self.get_joints(self.get_full_shape())

    def get_full_shape(
        self,
        betas: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        extra_betas: Optional[Tensor] = None,
    ):
        betas = self.betas if betas is None else betas
        if extra_betas is not None:
            betas = betas + extra_betas
        if self.use_smplx:
            expression = expression if expression is not None else self.expression
            shape_components = torch.cat([betas, expression], dim=-1)
        else:
            shape_components = betas
        if batch_size is not None:
            scale = int(batch_size / shape_components.shape[0])
            if scale > 1:
                shape_components = shape_components.expand(scale, -1)
        return shape_components

    def get_full_pose(
        self,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
    ):
        global_orient = global_orient if global_orient is not None else self.global_orient
        body_pose = body_pose if body_pose is not None else self.body_pose

        if self.use_smplx:
            left_hand_pose = left_hand_pose if left_hand_pose is not None else self.left_hand_pose
            right_hand_pose = right_hand_pose if right_hand_pose is not None else self.right_hand_pose
            jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
            leye_pose = leye_pose if leye_pose is not None else self.leye_pose
            reye_pose = reye_pose if reye_pose is not None else self.reye_pose

            if self.use_pca:
                left_hand_pose = torch.einsum('bi,ij->bj', [self.left_hand_pose, self.left_hand_components])
                right_hand_pose = torch.einsum('bi,ij->bj', [self.right_hand_pose, self.right_hand_components])

            full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                                body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
                                self.jaw_pose.reshape(-1, 1, 3),
                                self.leye_pose.reshape(-1, 1, 3),
                                self.reye_pose.reshape(-1, 1, 3),
                                left_hand_pose.reshape(-1, 15, 3),
                                right_hand_pose.reshape(-1, 15, 3)],
                                dim=1).reshape(-1, 165)

            # Add the mean pose of the model. Does not affect the body, only the
            # hands when flat_hand_mean == False
            full_pose += self.pose_mean

        else:
            full_pose = torch.cat([global_orient, body_pose], dim=1)  # [N, J*3]

        return full_pose

    def get_joints(self, betas: Tensor, return_shape_offsets: bool = False):
        # Get default params
        v_template = self.v_template
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1) if self.use_smplx else self.shapedirs
        J_regressor = self.J_regressor

        # Add shape contribution
        shape_offsets = blend_shapes(betas, shapedirs)
        v_shaped = v_template + shape_offsets

        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)

        if return_shape_offsets:
            return J, shape_offsets
        else:
            return J

    def get_full_transform(self, betas: Tensor, pose: Tensor, pose2rot: bool = True) -> dict:
        ''' 
            Parameters
            ----------
            betas : torch.tensor BxNB
                The tensor of shape parameters
            pose : torch.tensor Bx(J + 1) * 3
                The pose parameters in axis-angle format
            pose2rot: bool, optional
                Flag on whether to convert the input pose tensor to rotation
                matrices. The default value is True. If False, then the pose tensor
                should already contain rotation matrices and have a size of
                Bx(J + 1)x9

            Returns
            -------
            dict : str -> RigidTransform
        '''
        # Get default params
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1) if self.use_smplx else self.shapedirs
        shape_offsets = blend_shapes(betas, shapedirs)
        v_shaped = self.v_template + shape_offsets

        # Get the joints
        # NxJx3 array
        J = vertices2joints(self.J_regressor, v_shaped)
        transform_J_shape_offset = RigidTransform(T=J - self.J_template)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = rot_mats[:, 1:, :, :] - ident
        else:
            rot_mats = pose.view(batch_size, -1, 3, 3)
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), self.posedirs).view(batch_size, -1, 3)  # (N x P) x (P, V * 3) -> N x V x 3

        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        transform_V_shape_offset = RigidTransform(T=shape_offsets)
        transform_V_pose_offset = RigidTransform(T=pose_offsets)

        transform_J_pose_rigid = RigidTransform(SE3=A)
        transform_V_pose_rigid = RigidTransform(SE3=T)

        return {
            "V_shape_offset": transform_V_shape_offset,
            "V_pose_offset": transform_V_pose_offset,
            "V_pose_rigid": transform_V_pose_rigid,
            "J_shape_offset": transform_J_shape_offset,
            "J_pose_rigid": transform_J_pose_rigid,
        }

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        # Motion-X params
        flame_betas: Optional[Tensor] = None,
        flame_expression: Optional[Tensor] = None,
        # New
        extra_betas: Optional[Tensor] = None,
    ) -> Tuple[RigidTransform, RigidTransform]:
        """
        Input:
            body_pose: torch.Tensor, shape = (N, V*3) or (N, V, 3, 3)
            global_orient: torch.Tensor, shape = (N, 3) or (N, 3, 3)
            betas: torch.Tensor, shape = (N, 10)
            transl: torch.Tensor, shape = (N, 3)
        """
        full_shape = self.get_full_shape(betas=betas, expression=expression, extra_betas=extra_betas)
        
        full_pose = self.get_full_pose(
            body_pose=body_pose,
            global_orient=global_orient,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
        )

        transforms = self.get_full_transform(
            betas=full_shape,
            pose=full_pose,
            pose2rot=True,
        )

        transform_V = RigidTransform.compose(
            transforms['V_shape_offset'],
            transforms['V_pose_offset'],
            transforms['V_pose_rigid'],
        )

        transform_J = RigidTransform.compose(
            transforms['J_shape_offset'],
            transforms['J_pose_rigid'],
        )
        
        if transl is not None:
            transform_transl = RigidTransform(T=transl)
            transform_V = transform_V.compose(transform_transl)
            transform_J = transform_J.compose(transform_transl)
            transforms['G_transl_offset'] = transform_transl
        else:
            N = full_shape.shape[0]
            transforms['G_transl_offset'] = RigidTransform(
                SE3=torch.eye(4, 4, dtype=full_shape.dtype, device=full_shape.device).expand(N, 4, 4)
            )
        
        return transform_J, transform_V, transforms
