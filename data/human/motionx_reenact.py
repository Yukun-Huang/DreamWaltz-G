import numpy as np
import zipfile
from typing import Any
import json
import os
import cv2
import os.path as osp
from collections import defaultdict

from configs.paths import MOTIONX_REENACT_ROOT


class MotionX_ReEnact(object):
    def __init__(self, root: str = MOTIONX_REENACT_ROOT, from_zip: bool = True) -> None:
        dat = defaultdict(dict)
        filekeys = set()
        # Load from zip
        if from_zip:
            zip_file = osp.join(root, 'Motion-X-ReEnact.zip')
            assert osp.isfile(zip_file), f'File {zip_file} not found'
            archive = zipfile.ZipFile(zip_file, mode='r')
            for filepath in archive.namelist():
                filekey = None
                if filepath.startswith('video/') and filepath.endswith('mp4'):
                    filekey = osp.splitext(filepath.replace('video/', '', 1))[0]
                    dat['video'][filekey] = filepath
                elif filepath.startswith('inpainting_p3m_seg/') and filepath.endswith('mp4'):
                    filekey = osp.splitext(filepath.replace('inpainting_p3m_seg/', '', 1))[0]
                    dat['inpainting'][filekey] = filepath
                elif filepath.startswith('motion/') and filepath.endswith('json'):
                    filekey = osp.splitext(filepath.replace('motion/', '', 1))[0]
                    dat['motion'][filekey] = filepath
                if filekey is not None:
                    filekeys.add(filekey)
                # print(filepath)
            self.archive = archive
        else:
            raise NotImplementedError
        # Sanity check
        assert len(dat['video']) == len(dat['inpainting']) == len(dat['motion']) == len(filekeys)
        # Save
        self.dat = dat
        self.from_zip = from_zip
        self.filekeys = filekeys

    def parse_camera_params(self, camera_params) -> dict:
        # convert camera params: world_scale, cam_R, cam_T, intrins
        world_scale = camera_params['world_scale']

        extrinsic = np.eye(4).reshape(1, 4, 4).repeat(camera_params['cam_R'].shape[0], axis=0)  # [N, 4, 4]
        extrinsic[:, :3, :3] = camera_params['cam_R']
        extrinsic[:, :3, 3] = camera_params['cam_T']

        # flip y axis
        extrinsic[:, 1, :] *= -1

        intrinsics = np.zeros((camera_params['intrins'].shape[0], 3, 3))  # [N, 3, 3]
        fx, fy = camera_params['intrins'][:, 0], camera_params['intrins'][:, 1]  # [N], [N]
        cx, cy = camera_params['intrins'][:, 2], camera_params['intrins'][:, 3]  # [N], [N]
        
        x_sign = 1
        y_sign = -1

        intrinsics[:, 0, 0] = fx * x_sign
        intrinsics[:, 1, 1] = fy * y_sign
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        intrinsics[:, 2, 2] = 1

        image_width, image_height = int(cx[0] * 2), int(cy[0] * 2)
        
        fov_x = 2 * np.arctan(cx / fx)
        fov_y = 2 * np.arctan(cy / fy)
        tanfov_x = np.tan(fov_x / 2)  # [N]
        tanfov_y = np.tan(fov_y / 2)  # [N]

        aspect_ratio = abs((fx.mean() / fy.mean()).item())

        fov_x = np.rad2deg(fov_x)  # [N]
        fov_y = np.rad2deg(fov_y)  # [N]

        return {
            'extrinsic': extrinsic,
            'intrinsics': intrinsics,
            'image_height': image_height,
            'image_width': image_width,
            'tanfov_x': tanfov_x,
            'tanfov_y': tanfov_y,
            'fov_x': fov_x,
            'fov_y': fov_y,
            'aspect_ratio': aspect_ratio,
            'world_scale': world_scale,
        }

    def load_json_params(self, filepath:str):
        json_data = json.load(self.archive.open(filepath, mode='r'))
        image_info: list = json_data['images']
        anno_info: list = json_data['annotations']
        """
        Example of Image Info
        image_info[0] = {
            'id': 8356,
            'image_id': 6132,
            'file_name': 'Play_the_stringed_guqin_11_clip2/000001.png',
        }
        """
        # load and to numpy
        smplx_params = defaultdict(list)
        camera_params = defaultdict(list)
        for anno in anno_info:
            for k, v in anno['smplx_params'].items():
                smplx_params[k].append(v)
            for k, v in anno['cam_params'].items():
                camera_params[k].append(v)
        for k, v in smplx_params.items():
            smplx_params[k] = np.array(v)
        for k, v in camera_params.items():
            camera_params[k] = np.array(v)
        
        # convert camera params: world_scale, cam_R, cam_T, intrins
        camera_params = self.parse_camera_params(camera_params)

        # convert smplx params
        new_smplx_params = {
            'global_orient': smplx_params['root_orient'],  # controls the global root orientation, [N, 3]
            'body_pose': smplx_params['pose_body'],  # controls the body, [N, 63]
            'left_hand_pose': smplx_params['pose_hand'][:, :45],  # controls the finger articulation, [N, 45]
            'right_hand_pose': smplx_params['pose_hand'][:, 45:],  # controls the finger articulation, [N, 45]
            'jaw_pose': smplx_params['pose_jaw'],  # controls the yaw pose, [N, 3]
            # 'face_expr': smplx_params['face_expr'],  # controls the face expression, [N, 50]
            # 'face_shape': smplx_params['face_shape'],  # controls the face shape, [N, 100]
            'transl': smplx_params['trans'],  # controls the global body position, [N, 3]
            'betas': smplx_params['betas'],  # controls the body shape, [N, 10]
        }
        for k in new_smplx_params.keys():
            new_smplx_params[k] = new_smplx_params[k][np.newaxis, :, :]
        smplx_params = new_smplx_params

        # return
        return smplx_params, camera_params

    def load_video_frames(self, video_path:str):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames, fps, frame_count

    def extract_video(self, filename:str, save_path:str, video_type:str='video'):
        video_path = self.dat[video_type][filename]
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        video_stream = self.archive.read(video_path)
        with open(save_path, 'wb') as f:
            f.write(video_stream)

    def overlay_pngs_on_video(
        self,
        video_filename:str,
        image_folder:str,
        output_folder:str,
        video_type:str='inpainting',
        save_iamges:bool=True,
        save_video_frames:bool=False,
    ):
        """
        Overlay transparent PNG images from a folder onto a video and save the result.

        Parameters:
        - video_filename: Path to the input video file.
        - image_folder: Path to the folder containing PNG images with transparency.
        - output_folder: Path to save the output videos.
        - position: A tuple (x, y) indicating the position to overlay the PNG on the video frames.
        """
        # Make sure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get the paths to the video and inpainting video
        video_path = osp.join(output_folder, 'video.mp4')
        inpainting_path = osp.join(output_folder, 'inpainting.mp4')
        self.extract_video(video_filename, video_path, video_type='video')
        self.extract_video(video_filename, inpainting_path, video_type='inpainting')

        # Open the video
        if video_type == 'inpainting':
            frame_source = cv2.VideoCapture(inpainting_path)
        else:
            frame_source = cv2.VideoCapture(video_path)
        
        if save_video_frames:
            video_cap = cv2.VideoCapture(video_path)
        # inpainting_cap = cv2.VideoCapture(inpainting_path)

        fps = int(frame_source.get(cv2.CAP_PROP_FPS))
        frame_count = int(frame_source.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(frame_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(frame_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get list of PNG files sorted by name
        png_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        
        # Create a video writer
        output_path = osp.join(output_folder, 'overlay.mp4')
        try:
            from utils.video import VideoWriterPyAV
            out_video = VideoWriterPyAV(output_path, fps=fps, bit_rate=2000)
            use_pyav = True
            print('Using PyAV Video Writer!')
        except ImportError:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            use_pyav = False

        # Create image folders to save frames
        if save_iamges:
            out_overlay_folder = osp.join(output_folder, 'overlay_frames')
            out_video_folder = osp.join(output_folder, 'video_frames')
            os.makedirs(out_overlay_folder, exist_ok=True)
            if save_video_frames:
                os.makedirs(out_video_folder, exist_ok=True)

        frame_count = 0
        while True:
            ret, frame = frame_source.read()
            if not ret or frame_count >= len(png_files):
                break

            # Read the corresponding PNG image
            png_path = os.path.join(image_folder, png_files[frame_count])
            png_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)  # Read with alpha channel

            if png_img.shape[2] == 3:
                # Add alpha channel if missing
                png_img = np.dstack((png_img, np.full((png_img.shape[0], png_img.shape[1]), 255, dtype=np.uint8)))

            # Resize both images to the smaller size
            min_height = min(frame.shape[0], png_img.shape[0])
            min_width = min(frame.shape[1], png_img.shape[1])
            resized_frame = cv2.resize(frame, (min_width, min_height))
            resized_png_img = cv2.resize(png_img, (min_width, min_height))

            # Extract the alpha channel from the PNG image
            alpha_channel = resized_png_img[:, :, 3] / 255.0
            color_channels = resized_png_img[:, :, :3]

            # Blend the PNG image with the frame
            for c in range(3):
                resized_frame[:, :, c] = (alpha_channel * color_channels[:, :, c] +
                                        (1 - alpha_channel) * resized_frame[:, :, c])

            if save_iamges:
                cv2.imwrite(osp.join(out_overlay_folder, f'{frame_count:06d}.png'), resized_frame)
                if save_video_frames:
                    _, raw_frame = video_cap.read()
                    raw_frame = cv2.resize(raw_frame, (min_width, min_height))
                    cv2.imwrite(osp.join(out_video_folder, f'{frame_count:06d}.png'), raw_frame)

            if use_pyav:
                out_video.write(resized_frame[:, :, ::-1])  # BGR -> RGB
            else:
                out_video.write(resized_frame)
            frame_count += 1

        frame_source.release()
        out_video.release()
        if save_video_frames:
            video_cap.release()
        # inpainting_cap.release()

    def get_smpl_params(self, filename:str, model_type:str='smplx', fps:int=None, stand_fps:int=None):
        assert model_type == 'smplx'

        # filename = "music/Play_the_stringed_guqin_11_clip2"
        json_path = self.dat['motion'][filename]
        
        smplx_params, camera_params = self.load_json_params(json_path)
        num_frames = smplx_params['body_pose'].shape[1] 

        if fps is not None and stand_fps is not None:
            fps_step = np.ceil(fps / stand_fps)
            slected_frames = [i for i in range(num_frames) if i % fps_step == 0]
            for k in smplx_params.keys():
                smplx_params[k] = smplx_params[k][:, slected_frames, :]

        return smplx_params, camera_params


""" All Files:
inpainting_p3m_seg/animation/
inpainting_p3m_seg/animation/Ways_to_Catch_360_clip1.mp4
inpainting_p3m_seg/animation/Ways_to_Catch_Between_the_Legs_clip1.mp4
inpainting_p3m_seg/animation/Ways_to_Catch_Large_Ball_clip1.mp4
inpainting_p3m_seg/animation/Ways_to_Jump_+_Sit_+_Fall_Broken_Ankle_clip1.mp4
inpainting_p3m_seg/animation/Ways_to_Jump_+_Sit_+_Fall_Fun_Photo_clip1.mp4
inpainting_p3m_seg/fitness/
inpainting_p3m_seg/fitness/_BURPEES_clip4.mp4
inpainting_p3m_seg/fitness/DIAMOND_LEG_ROTATION_clip5.mp4
inpainting_p3m_seg/haa500/
inpainting_p3m_seg/haa500/baseball_pitch_11_clip1.mp4
inpainting_p3m_seg/haa500/basketball_dribble_4_clip1.mp4
inpainting_p3m_seg/haa500/basketball_shoot_16_clip1.mp4
inpainting_p3m_seg/humman/
inpainting_p3m_seg/humman/After_standing_leg_lifts_R_1_clip1.mp4
inpainting_p3m_seg/kungfu/
inpainting_p3m_seg/kungfu/Aerial_Kick_Kungfu_wushu_14_clip2.mp4
inpainting_p3m_seg/kungfu/Aerial_Kick_Kungfu_wushu_21_clip1.mp4
inpainting_p3m_seg/kungfu/Shaolin_KungFu_Staff_Workout_Training_3_clip2.mp4
inpainting_p3m_seg/kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step15_clip1.mp4
inpainting_p3m_seg/kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step9_clip1.mp4
inpainting_p3m_seg/music/
inpainting_p3m_seg/music/Play_the_stringed_guqin_11_clip2.mp4
inpainting_p3m_seg/perform/
inpainting_p3m_seg/perform/eye_training_clip3.mp4
inpainting_p3m_seg/perform/peking_opera_performance_man_clip3.mp4
motion/animation/
motion/animation/Ways_to_Catch_360_clip1.json
motion/animation/Ways_to_Catch_Between_the_Legs_clip1.json
motion/animation/Ways_to_Catch_Large_Ball_clip1.json
motion/animation/Ways_to_Jump_+_Sit_+_Fall_Broken_Ankle_clip1.json
motion/animation/Ways_to_Jump_+_Sit_+_Fall_Fun_Photo_clip1.json
motion/fitness/
motion/fitness/_BURPEES_clip4.json
motion/fitness/ALT_TRICEP_KICKBACKS_clip6.json
motion/fitness/DIAMOND_LEG_ROTATION_clip5.json
motion/haa500/
motion/haa500/baseball_pitch_11_clip1.json
motion/haa500/basketball_dribble_4_clip1.json
motion/haa500/basketball_shoot_16_clip1.json
motion/humman/
motion/humman/After_standing_leg_lifts_R_1_clip1.json
motion/kungfu/
motion/kungfu/Aerial_Kick_Kungfu_wushu_14_clip2.json
motion/kungfu/Aerial_Kick_Kungfu_wushu_21_clip1.json
motion/kungfu/Shaolin_KungFu_Staff_Workout_Training_3_clip2.json
motion/kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step15_clip1.json
motion/kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step9_clip1.json
motion/music/
motion/music/Play_the_stringed_guqin_11_clip2.json
motion/perform/
motion/perform/eye_training_clip3.json
motion/perform/peking_opera_performance_man_clip3.json
video/animation/
video/animation/Ways_to_Catch_360_clip1.mp4
video/animation/Ways_to_Catch_Between_the_Legs_clip1.mp4
video/animation/Ways_to_Catch_Large_Ball_clip1.mp4
video/animation/Ways_to_Jump_+_Sit_+_Fall_Broken_Ankle_clip1.mp4
video/animation/Ways_to_Jump_+_Sit_+_Fall_Fun_Photo_clip1.mp4
video/fitness/
video/fitness/_BURPEES_clip4.mp4
video/fitness/ALT_TRICEP_KICKBACKS_clip6.mp4
video/fitness/DIAMOND_LEG_ROTATION_clip5.mp4
video/haa500/
video/haa500/baseball_pitch_11_clip1.mp4
video/haa500/basketball_dribble_4_clip1.mp4
video/haa500/basketball_shoot_16_clip1.mp4
video/humman/
video/humman/After_standing_leg_lifts_R_1_clip1.mp4
video/kungfu/
video/kungfu/Aerial_Kick_Kungfu_wushu_14_clip2.mp4
video/kungfu/Aerial_Kick_Kungfu_wushu_21_clip1.mp4
video/kungfu/Shaolin_KungFu_Staff_Workout_Training_3_clip2.mp4
video/kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step15_clip1.mp4
video/kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step9_clip1.mp4
video/music/
video/music/Play_the_stringed_guqin_11_clip2.mp4
video/perform/
video/perform/eye_training_clip3.mp4
video/perform/peking_opera_performance_man_clip3.mp4
"""
