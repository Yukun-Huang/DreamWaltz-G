import torch
import numpy as np
from typing import Iterable, Optional, Union, Any
from .pw3d import PW3D
from .amass import AMASS
from .aist import AIST
from .hybrik import Hybrik
from .demo import Demo
from .motionx import MotionX
from .motionx_reenact import MotionX_ReEnact
from .talkshow import TalkShow
from core.human.smpl_model import SMPLModel


DATASET_CARDS = {
    'demo': Demo,
    '3dpw': PW3D,
    'amass': AMASS,
    'aist': AIST,
    'hybrik': Hybrik,
    'motionx': MotionX,
    'motionx_reenact': MotionX_ReEnact,
    'talkshow': TalkShow,
}


def expand_humans(smpl_params, num_person, device):
    # Expand to Multiple Person
    for k in smpl_params.keys():
        if isinstance(smpl_params[k], torch.Tensor):
            smpl_params[k] = smpl_params[k].expand(num_person, -1)
    smpl_params['transl'] = get_transl_pattern(num_person, device=device)
    return smpl_params


def get_transl_pattern(num_person, device, spacing=0.8):
    if num_person <= 1:
        return None
    transl_pattern = {
        2: torch.tensor([[-spacing, 0.0, 0.0], [+spacing, 0.0, 0.0]]),
        3: torch.tensor([[0.0, 0.0, +spacing], [-spacing, 0.0, 0.0], [+spacing, 0.0, 0.0]]),
        4: torch.tensor([[+spacing, 0.0, +spacing], [+spacing, 0.0, -spacing],
                            [-spacing, 0.0, +spacing], [-spacing, 0.0, -spacing]]),
        5: torch.tensor([[+spacing, 0.0, +spacing], [+spacing, 0.0, -spacing], [0.0, 0.0, 0.0],
                            [-spacing, 0.0, +spacing], [-spacing, 0.0, -spacing]]),
    }
    return transl_pattern[num_person].to(device)


def load_smpl_sequences(scene: str, model_type: str, camera_sequences: Optional[dict] = None, _dataset: Any = None, **kwargs):
    """
        scene (str): Motion source, e.g., "3dpw,dance", "3dpw,dance,200-275", "3dpw,dance,200-275-5".
    """
    dataset, filename, *frame_args = scene.split(',')
    if len(frame_args) == 1:
        frame_args = tuple(map(int, frame_args[0].split('-')))
        if len(frame_args) == 2:
            kwargs['frame_range'] = (frame_args[0], frame_args[1])
        else:
            assert len(frame_args) == 3, f"Invalid Scene Format: {scene}"
            assert 'frame_interval' not in kwargs or kwargs['frame_interval'] is None, "Frame Interval Already Specified!"
            kwargs['frame_range'] = (frame_args[0], frame_args[1])
            kwargs['frame_interval'] = frame_args[2]
    else:
        assert len(frame_args) == 0, f"Invalid Scene Format: {scene}"
    # Load Data
    if _dataset is None:
        _dataset = DATASET_CARDS[dataset]()
    if dataset != 'motionx_reenact':
        smpl_seqs = _dataset.get_smpl_params(filename, model_type=model_type)
    else:
        smpl_seqs, cam_seqs = _dataset.get_smpl_params(filename, model_type=model_type)
        if camera_sequences is not None:
            camera_sequences.update(cam_seqs)
    # Preprocess SMPL Seqs
    smpl_seqs = preprocess_smpl_sequences(smpl_seqs, dataset=dataset, **kwargs)
    num_person, num_frame, *_ = smpl_seqs['body_pose'].shape
    # Return
    return smpl_seqs, num_person, num_frame


def preprocess_smpl_sequences(
    smpl_seqs: dict,
    dataset: str,
    frame_range: Optional[Iterable[int]] = None,
    frame_interval: Optional[int] = None,
    device: Optional[torch.device] = None,
    num_person: Optional[int] = None,
    person_indices: Optional[Iterable] = None,
    to_tensor: bool = True,
    pop_betas: bool = False,
    pop_transl: bool = False,
    centralize_pelvis: bool = True,
    pop_global_orient: bool = False,
    normalize_transl: bool = False,
    smpl_model: Optional[SMPLModel] = None,
):
    if num_person is not None or person_indices is not None:
        if person_indices is None and num_person is not None:
            person_indices = [i for i in range(num_person)]
        for k in smpl_seqs.keys():
            smpl_seqs[k] = smpl_seqs[k][person_indices, ...]
    
    if frame_range is not None or frame_interval is not None:
        if frame_range is None:
            frame_range = (0, smpl_seqs['body_pose'].shape[1])
        if frame_interval is None:
            frame_interval = 1
        for k in smpl_seqs.keys():
            if smpl_seqs[k].ndim < 3:  # betas
                continue
            smpl_seqs[k] = smpl_seqs[k][:, range(*frame_range, frame_interval), :]

    # adjust shape
    if 'betas' in smpl_seqs:
        if pop_betas:
            smpl_seqs.pop('betas')
        else:
            smpl_seqs['betas'] = smpl_seqs['betas'][..., :smpl_model.model.num_betas]

    # adjust global_orient
    if 'global_orient' in smpl_seqs and pop_global_orient:
        smpl_seqs.pop('global_orient')
    
    # adjust translation
    if 'transl' in smpl_seqs:
        if pop_transl:
            smpl_seqs.pop('transl')
        elif normalize_transl:
                # transl_mean = np.mean(smpl_seqs['transl'], axis=(0, 1), keepdims=True)
                transl_mean = np.mean(smpl_seqs['transl'], axis=(0, ), keepdims=True)
                smpl_seqs['transl'] -= transl_mean

    if centralize_pelvis:
        transl_offset = smpl_model.pelvis_position.detach().cpu().numpy()
        if 'transl' in smpl_seqs:
            smpl_seqs['transl'] -= transl_offset[np.newaxis, ...]
        else:
            n_person, n_frame, _ = smpl_seqs['body_pose'].shape
            smpl_seqs['transl'] = - np.broadcast_to(transl_offset, (n_person, n_frame, 3))

    if to_tensor:
        for k in smpl_seqs:
            smpl_seqs[k] = torch.tensor(smpl_seqs[k], dtype=torch.float, device=device)

    # adjust hand pose
    if dataset == 'talkshow' and (smpl_seqs['left_hand_pose'].shape[-1] != 45 or smpl_seqs['right_hand_pose'].shape[-1] != 45):
        assert to_tensor
        left_hand_pose = smpl_seqs.pop('left_hand_pose')  # (1, 300, 12)
        right_hand_pose = smpl_seqs.pop('right_hand_pose')  # (1, 300, 12)
        left_hand_components = smpl_model.model.left_hand_components
        right_hand_components = smpl_model.model.right_hand_components
        smpl_seqs['left_hand_pose'] =  torch.einsum('nti,ij->ntj', [left_hand_pose, left_hand_components])
        smpl_seqs['right_hand_pose'] = torch.einsum('nti,ij->ntj', [right_hand_pose, right_hand_components])

    # if dataset == '3dpw':
    #     assert 'left_hand_pose' not in smpl_seqs and 'right_hand_pose' not in smpl_seqs
    #     smpl_seqs['left_hand_pose'] = torch.zeros((smpl_seqs['body_pose'].shape[0], smpl_seqs['body_pose'].shape[1], 45), dtype=torch.float, device=device)
    #     smpl_seqs['right_hand_pose'] = torch.zeros((smpl_seqs['body_pose'].shape[0], smpl_seqs['body_pose'].shape[1], 45), dtype=torch.float, device=device)
    #     if not smpl_model.model.flat_hand_mean:
    #         smpl_seqs['left_hand_pose'] -= smpl_model.model.left_hand_mean.unsqueeze(0).unsqueeze(0).to(device)
    #         smpl_seqs['right_hand_pose'] -= smpl_model.model.right_hand_mean.unsqueeze(0).unsqueeze(0).to(device)
    
    # 'global_orient': global_orient,  # np.ndarray/torch.Tensor, (2, F, 3)
    # 'body_pose': body_pose,          # np.ndarray/torch.Tensor, (2, F, 63/69)
    # 'betas': betas,                  # np.ndarray/torch.Tensor, (2, F, 10)
    # 'transl': transl,                # np.ndarray/torch.Tensor, (2, F, 3)
    # ...
    return smpl_seqs
