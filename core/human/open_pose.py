# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.
import math
from typing import Callable, List, NamedTuple, Tuple, Union
import cv2
import numpy as np

eps = 0.01

ALPHA_MODE = True


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1
    dist: float = -1.0


HandResult = List[Keypoint]
FaceResult = List[Keypoint]


class BodyResult(NamedTuple):
    # Note: Using `Union` instead of `|` operator as the ladder is a Python
    # 3.10 feature.
    # Annotator code should be Python 3.8 Compatible, as controlnet repo uses
    # Python 3.8 environment.
    # https://github.com/lllyasviel/ControlNet/blob/d3284fcd0972c510635a4f5abe2eeb71dc0de524/environment.yaml#L6
    keypoints: List[Union[Keypoint, None]]
    total_score: float
    total_parts: int


class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]
    face: Union[FaceResult, None]


def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint], radius: int = 4, stickwidth: int = 4, flip_LR: bool = False) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    """
        1: noise
        2: neck
        3: right_shoulder
        4: right_elbow
        5: right_wrist
        6: left_shoulder
        7: left_elbow
        8: left_wrist
        9: right_hip
        10: right_knee
        11: right_ankle
        12: left_hip
        13: left_knee
        14: left_ankle
        15: right_eye
        16: left_eye
        17: right_ear
        18: left_ear
    """
    if flip_LR:
        keypoints = [
            keypoints[0], keypoints[1],
            keypoints[5], keypoints[6], keypoints[7],
            keypoints[2], keypoints[3], keypoints[4],
            keypoints[11], keypoints[12], keypoints[13],
            keypoints[8], keypoints[9], keypoints[10],
            keypoints[15], keypoints[14],
            keypoints[17], keypoints[16],
        ]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85],
    ]

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x = int(keypoint.x * W)
        y = int(keypoint.y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), radius, color, thickness=-1)

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)

        if ALPHA_MODE:
            cur_canvas = canvas.copy()
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        else:
            cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    return canvas


def draw_handpose(
    canvas: np.ndarray,
    keypoints: Union[List[Keypoint], None],
    radius: int = 4,
    thickness: int = 2,
    adaptive_dist_thres: float = None,
) -> np.ndarray:
    import matplotlib
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for keypoint in keypoints:
        if keypoint is None:
            continue
        x, y = int(keypoint.x * W), int(keypoint.y * H)
        if adaptive_dist_thres is not None:
            d = keypoint.dist
            assert d > 0.0
            r = min(adaptive_dist_thres / d, 1.0)
            this_radius = max(int(radius * r), 1)
        else:
            this_radius = radius
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), this_radius, (0, 0, 255), thickness=-1)
    
    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue

        x1, y1 = int(k1.x * W), int(k1.y * H)
        x2, y2 = int(k2.x * W), int(k2.y * H)

        if adaptive_dist_thres is not None:
            d1, d2 = k1.dist, k2.dist
            assert d1 > 0.0 and d2 > 0.0
            d = (d1 + d2) / 2
            r = min(adaptive_dist_thres / d, 1.0)
            this_thickness = max(int(thickness * r), 1)
        else:
            this_thickness = thickness

        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255
            cv2.line(canvas, (x1, y1), (x2, y2), color=color, thickness=this_thickness)

            if ALPHA_MODE:
                cur_canvas = canvas.copy()
                cv2.line(cur_canvas, (x1, y1), (x2, y2), color=color, thickness=this_thickness)
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            else:
                cv2.line(canvas, (x1, y1), (x2, y2), color=color, thickness=this_thickness)

    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None], radius: int = 3) -> np.ndarray:
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape
    for keypoint in keypoints:
        if keypoint is None:
            continue
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), radius, (255, 255, 255), thickness=-1)
    return canvas


def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand)
            canvas = draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = draw_facepose(canvas, pose.face)

    return canvas


def adaptive_draw_poses(
    poses: List[PoseResult],
    H,
    W,
    draw_body=True,
    draw_hand=True,
    draw_face=True,
    hand_dist_thres=None,
    flip_LR: bool = False,
):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    body_radius = 4
    body_stickwidth = 4
    hand_radius = 4
    hand_thickness = 2
    face_radius = 3

    if H != 512 or W != 512:
        r = (H + W) / 2. / 512.
        body_radius = max(int(body_radius * r), 1)
        body_stickwidth = max(int(body_stickwidth * r), 1)
        hand_radius = max(int(hand_radius * r), 1)
        hand_thickness = max(int(hand_thickness * r), 1)
        face_radius = max(int(face_radius * r), 1)

    for pose in poses:
        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints, radius=body_radius, stickwidth=body_stickwidth, flip_LR=flip_LR)

        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand, radius=hand_radius, thickness=hand_thickness, adaptive_dist_thres=hand_dist_thres)
            canvas = draw_handpose(canvas, pose.right_hand, radius=hand_radius, thickness=hand_thickness, adaptive_dist_thres=hand_dist_thres)

        if draw_face:
            canvas = draw_facepose(canvas, pose.face, radius=face_radius)

    return canvas
