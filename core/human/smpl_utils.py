import os.path as osp
import numpy as np
import pickle
from human_body_prior.models.vposer_model import VPoser
from configs.paths import HUMAN_TEMPLATES as MODEL_ROOT

VPOSER_PATH = osp.join(MODEL_ROOT, 'vposer/v2.0')
SELF_CONTACT_PATH = osp.join(MODEL_ROOT, 'selfcontact')


def build_human_body_prior(model_path=VPOSER_PATH) -> VPoser:
    from human_body_prior.tools.model_loader import load_model
    vp, vp_cfg = load_model(model_path, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
    return vp


def build_selfcontact(model_type='smplx', essentials_folder=SELF_CONTACT_PATH):
    from selfcontact import SelfContact
    assert model_type in ('smpl', 'smplx')
    return SelfContact( 
        essentials_folder=essentials_folder,
        geothres=0.3, 
        euclthres=0.02, 
        model_type=model_type,
        test_segments=True,
        compute_hd=False
    )


def watertight_smplx(vertices, faces, essentials_folder=SELF_CONTACT_PATH):
    inner_mouth_verts_path = osp.join(essentials_folder, 'segments/smplx/smplx_inner_mouth_bounds.pkl')
    vert_ids_wt = np.array(pickle.load(open(inner_mouth_verts_path, 'rb')))

    faces_wt = [[vert_ids_wt[i+1], vert_ids_wt[i], faces.max().item()+1] for i in range(len(vert_ids_wt)-1)]
    faces_wt = np.array(faces_wt, dtype=faces.dtype)
    faces_wt = np.concatenate((faces, faces_wt), axis=0)

    mouth_vert = np.mean(vertices[:, vert_ids_wt, :], axis=1, keepdims=True)
    vertices_wt = np.concatenate((vertices, mouth_vert), 1)

    return vertices_wt, faces_wt


OPENPOSE_KEYPOINT_NAMES = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'right_hip': 8,
    'right_knee': 9,
    'right_ankle': 10,
    'left_hip': 11,
    'left_knee': 12,
    'left_ankle': 13,
    'right_eye': 14,
    'left_eye': 15,
    'right_ear': 16,
    'left_ear': 17,
    # smplx only
    'left_wrist_new': 18,
    'left_middle1': 27,
    'left_middle2': 28,
    'left_middle3': 29,
    'left_middle': 30,
    'right_wrist_new': 39,
    'right_middle1': 48,
    'right_middle2': 49,
    'right_middle3': 50,
    'right_middle': 51,
}


def smpl_to_openpose(model_type='smplx', openpose_format='coco18', use_hands=True, use_face=True, use_face_contour=True):
    # https://github.com/vchoutas/smplify-x/blob/3e11ff1daed20c88cd00239abf5b9fc7ba856bb6/smplifyx/utils.py#L96
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    if openpose_format == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour, dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
    elif openpose_format == 'coco18':
        if model_type == 'smpl':
            return np.array([
                24, 12,          # nose, neck
                17, 19, 21,      # right_shoulder, right_elbow, right_wrist
                16, 18, 20,      # left_shoulder, left_elbow, left_wrist
                2, 5, 8,         # right_hip, right_knee, right_ankle
                1, 4, 7,         # left_hip, left_knee, left_ankle
                25, 26, 27, 28,  # right_eye, left_eye, right_ear, left_ear
            ], dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        elif model_type == 'smplx':
            body_mapping = np.array([
                55, 12,          # nose, neck
                17, 19, 21,      # right_shoulder, right_elbow, right_wrist
                16, 18, 20,      # left_shoulder, left_elbow, left_wrist
                2, 5, 8,         # right_hip, right_knee, right_ankle
                1, 4, 7,         # left_hip, left_knee, left_ankle
                56, 57, 58, 59,  # right_eye, left_eye, right_ear, left_ear
            ], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([
                    20,              # left_wrist
                    37, 38, 39, 66,  # left_thumb1, left_thumb2, left_thumb3, left_thumb
                    25, 26, 27, 67,  # left_index1, left_index2, left_index3, left_index
                    28, 29, 30, 68,  # left_middle1, left_middle2, left_middle3, left_middle
                    34, 35, 36, 69,  # left_ring1, left_ring2, left_ring3, left_ring
                    31, 32, 33, 70,  # left_pinky1, left_pinky2, left_pinky3, left_pinky
                ], dtype=np.int32)
                rhand_mapping = np.array([
                    21,              # right_wrist
                    52, 53, 54, 71,  # right_thumb1, right_thumb2, right_thumb3, right_thumb
                    40, 41, 42, 72,  # right_index1, right_index2, right_index3, right_index
                    43, 44, 45, 73,  # right_middle1, right_middle2, right_middle3, right_middle
                    49, 50, 51, 74,  # right_ring1, right_ring2, right_ring3, right_ring
                    46, 47, 48, 75,  # right_pinky1, right_pinky2, right_pinky3, right_pinky
                ], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(76, 76 + 51 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
