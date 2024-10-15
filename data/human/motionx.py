import torch
import numpy as np
import zipfile
from typing import Any
import os
import os.path as osp
import pickle as pkl
from glob import glob
from typing import Iterable
from collections import defaultdict

from configs.paths import MOTIONX_ROOT


class MotionX(object):
    def __init__(self, root: str = MOTIONX_ROOT, from_zip: bool = True) -> None:
        dat = defaultdict(dict)
        if from_zip:
            motion_zip_file = osp.join(root, 'motionx_smplx.zip')
            assert osp.isfile(motion_zip_file), motion_zip_file
            import zipfile
            archive = zipfile.ZipFile(motion_zip_file, mode='r')
            for filepath in archive.namelist():
                if filepath.endswith('.npy'):
                    fileattrs = filepath.split('/')
                    assert len(fileattrs) == 5
                    assert fileattrs[0] == 'motion_data'
                    assert fileattrs[1] == 'smplx_322'
                    dataset = fileattrs[2]
                    subset = fileattrs[3]
                    filename = fileattrs[4]
                    dat[dataset][f'{subset}/{osp.splitext(filename)[0]}'] = filepath
            self.archive = archive
        else:
            raise NotImplementedError
        self.dat = dat
        self.from_zip = from_zip
        self.dat_keys = ['perform', 'music', 'idea400', 'animation', 'fitness', 'humman', 'game_motion', 'aist', 'HAA500', 'kungfu', 'dance']

    def _load_fast_demo(self, demo_path='./assets/motionx_demo.npy'):
        smplx_params = np.load(demo_path)
        return {
            'jaw_pose': smplx_params[np.newaxis, :, 0:3],               # 3
            'global_orient': smplx_params[np.newaxis, :, 9:12],         # 3
            'body_pose': smplx_params[np.newaxis, :, 12:75],            # 63
            'left_hand_pose': smplx_params[np.newaxis, :, 75:120],      # 45
            'right_hand_pose': smplx_params[np.newaxis, :, 120:165],    # 45
            'expression': smplx_params[np.newaxis, :, 165:265],         # 100
        }

    def _load_file(self, filepath:str):
        motion = np.load(self.archive.open(filepath, mode='r'))
        return {
            'global_orient': motion[np.newaxis, :, 0:0+3],  # controls the global root orientation
            'body_pose': motion[np.newaxis, :, 3:3+63],  # controls the body
            'left_hand_pose': motion[np.newaxis, :, 66:66+45],  # controls the finger articulation
            'right_hand_pose': motion[np.newaxis, :, 111:111+45],  # controls the finger articulation
            'jaw_pose': motion[np.newaxis, :, 156:156+3],  # controls the yaw pose
            # 'face_expr': motion[np.newaxis, :, 159:159+50],  # controls the face expression
            # 'face_shape': motion[np.newaxis, :, 209:209+100],  # controls the face shape
            'flame_betas': motion[np.newaxis, :, 159:159+50],  # controls the face expression
            'flame_expression': motion[np.newaxis, :, 209:209+100],  # controls the face shape
            'transl': motion[np.newaxis, :, 309:309+3],  # controls the global body position
            'betas': motion[np.newaxis, :, 312:],  # controls the body shape. Body shape is static
        }

    def get_smpl_params(self, filename:str, model_type:str='smplx', fps:int=None, stand_fps:int=None):
        assert model_type == 'smplx'

        dataset, filedir = filename.split('/', maxsplit=1)  # e.g., filename = "aist/subset_0008/Dance_Pop_Hand_Wave"

        dat = self._load_file(self.dat[dataset][filedir])

        num_persons, num_frames, _ = dat['body_pose'].shape

        if fps is not None and stand_fps is not None:
            fps_step = np.ceil(fps / stand_fps)
            slected_frames = [i for i in range(num_frames) if i % fps_step == 0]
            for k in dat.keys():
                dat[k] = dat[k][:, slected_frames, :]

        # read text labels
        # semantic_text = np.loadtxt('semantic_labels/000001.npy')     # semantic labels

        return dat


if __name__ == '__main__':
    motionx = MotionX()

    filekeys = []

    for subset in motionx.dat.keys():
        for key in motionx.dat[subset].keys():
            filekeys.append(f'{subset}/{key}')

    print(len(filekeys))


"""
data['dance'].keys() = [
    'subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of',
    'subset_0000/A_Han_And_Tang_Dance_That_You_Will_Never_Get_Tired_Of_clip_1',
    'subset_0000/A_Hundred_Dances',
    'subset_0000/A_Hundred_Dances_clip_1',
    'subset_0000/A_Hundred_Dances_clip_2',
    'subset_0000/Adhere_To_The_First_Day_Of_Daily_Change_Flower',
    'subset_0000/Adhere_To_The_First_Day_Of_Daily_Change_Flower_clip_1',
    'subset_0000/Adhere_To_The_First_Day_Of_Daily_Change_Flower_clip_2',
    'subset_0000/All_Night_Gidle_Last_Bus',
    'subset_0000/All_Night_Gidle_Last_Bus_clip_1',
    'subset_0000/Allergy_Gidle',
    'subset_0000/Allergy_Gidle_clip_1',
    'subset_0000/Allergy_Gidle_clip_2',
    'subset_0000/Apink_Mr_Chu',
    'subset_0000/Archeology_Sending_The_Bright_Moon_Dance_New_Chinese_Love',
    'subset_0000/Archeology_Sending_The_Bright_Moon_Dance_New_Chinese_Love_clip_1',
    'subset_0000/Arrangement_Easy-to-learn_Dance_Clips_Moonlight_In_The_Lotus_Pond',
    'subset_0000/Arrangement_Easy-to-learn_Dance_Clips_Moonlight_In_The_Lotus_Pond_clip_1',
    'subset_0000/As_Expected_Of_Ms',
    'subset_0000/As_Expected_Of_Ms_clip_1',
    'subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It',
    'subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1',
    'subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1_clip_1',
    'subset_0000/Bai_Jingting_Said_It_Looks_Good_And_Then_I_Posted_It_Clip1_clip_2',
    'subset_0000/Ballet_Spin_Compilation',
    'subset_0000/Ballet_Spin_Compilation_clip_1',
    'subset_0000/Basic_Dai_Dance',
    'subset_0000/Basic_Dai_Dance_clip_1',
    'subset_0000/Basic_Skills_Practice',
    'subset_0000/Basic_Skills_Practice_clip_1',
    'subset_0000/Basic_Skills_Practice_clip_2',
    'subset_0000/Basic_Skills_Practice_clip_3',
    'subset_0001/Dai',
    'subset_0001/Dance_Long_Moon_Emeritus',
    'subset_0001/Dance_Moon_Dancer',
    'subset_0001/Dance_Moon_Dancer_clip_1',
    'subset_0001/Dancing_Daily_Metoo_Zhang_Yuanying',
    'subset_0001/Dancing_Daily_Metoo_Zhang_Yuanying_clip_1',
    'subset_0001/Dancing_Everyday_I_Am_Ive',
    'subset_0001/Deep_Love_And_Rain_Classical_Dance',
    'subset_0001/Deep_Love_And_Rain_Classical_Dance_clip_1',
    'subset_0001/Deep_Love_And_Rain_Classical_Dance_clip_2',
    'subset_0001/Electrical_Version_Babymonster_Flip',
    'subset_0001/Electrical_Version_Babymonster_Flip_clip_1',
    'subset_0001/Electrical_Version_Babymonster_Flip_clip_2',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_1',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_2',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_3',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_4',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_5',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_6',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_7',
    'subset_0001/Electrician_Version_Aoa_Catwalk_Lightly_Flip_clip_8',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump_clip_1',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump_clip_2',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump_clip_3',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump_clip_4',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump_clip_5',
    'subset_0001/Electrician_Version_Exo_The_Eve_Jump_clip_6',
    'subset_0001/Electrician_Version_Jisoo_Flower_Jump',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_1',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_2',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_3',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_4',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_5',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_6',
    'subset_0001/Electrician_Version_Love_Shot_Dormitory_Million_Lights_clip_7',
    'subset_0001/Electrician_Version_Rikimaru_Up_And_Down_Jump',
    'subset_0001/Electrician_Version_Rikimaru_Up_And_Down_Jump_clip_1',
    'subset_0001/Electrician_Version_Rikimaru_Up_And_Down_Jump_clip_2',
    'subset_0001/Electrician_Version_Sunmi_Tail_Jump',
    'subset_0001/Electrician_Version_Sunmi_Tail_Jump_clip_1',
    'subset_0001/Electrician_Version_Sunmi_Tail_Jump_clip_2',
    'subset_0001/Electrician_Version_Sunmi_Tail_Jump_clip_3',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip_clip_1',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip_clip_2',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip_clip_3',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip_clip_4',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip_clip_5',
    'subset_0001/Electrician_Version_Tell_Me_Tell_Me_Flip_clip_6',
    'subset_0001/Electrician_Version_Twice_Go_Hard_Jump',
    'subset_0001/Electrician_Version_Twice_Go_Hard_Jump_clip_1',
    'subset_0001/Electrician_Version_Twice_Go_Hard_Jump_clip_2',
    'subset_0001/Every_Body_Has_A_Unique_Charm',
    'subset_0001/Every_Body_Has_A_Unique_Charm_clip_1',
    'subset_0002/King_Kong_Exotic_Dance',
    'subset_0002/King_Kong_Exotic_Dance_Teaching',
    'subset_0002/King_Kong_Exotic_Dance_Teaching_clip_1',
    'subset_0002/King_Kong_Exotic_Dance_Teaching_clip_2',
    'subset_0002/King_Kong_Exotic_Dance_clip_1',
    'subset_0002/King_Kong_Exotic_Dance_clip_2',
    'subset_0002/King_Kong_Pillow_Tale',
    'subset_0002/King_Kong_Pillow_Tale_Clip1',
    'subset_0002/King_Kong_Pillow_Tale_Clip1_clip_1',
    'subset_0002/King_Kong_Pillow_Tale_Clip1_clip_2',
    'subset_0002/King_Kong_Pillow_Tale_Clip1_clip_3',
    'subset_0002/King_Kong_Taihu_Beauty_Clip1',
    'subset_0002/King_Kong_Taihu_Beauty_Clip1_clip_1',
    'subset_0002/King_Kong_Taihu_Beauty_Clip1_clip_2',
    'subset_0002/King_Kong_Taihu_Beauty_Clip1_clip_3',
    'subset_0002/Night_In_Ulaanbaatar_Modern_Mongolian_Dance',
    'subset_0002/Night_In_Ulaanbaatar_Modern_Mongolian_Dance_clip_1',
    'subset_0002/North_Dance_Center_Of_Gravity_Training_Combination',
    'subset_0002/North_Dance_Center_Of_Gravity_Training_Combination_clip_1',
    'subset_0002/Queencard_Shines_Brightly_From_Head_To_Toe',
    'subset_0002/Queencard_Shines_Brightly_From_Head_To_Toe_clip_1',
    'subset_0002/Queencard_Shines_Brightly_From_Head_To_Toe_clip_2',
    'subset_0002/Salary_Arrives',
    'subset_0002/Salary_Arrives_clip_1',
    'subset_0002/Say_No_To_Appearance_Anxiety_Allergy_G_I_Dle',
    'subset_0002/Say_No_To_Appearance_Anxiety_Allergy_G_I_Dle_clip_1',
    'subset_0002/Say_No_To_Appearance_Anxiety_Allergy_G_I_Dle_clip_2',
    'subset_0002/Say_No_To_Appearance_Anxiety_Allergy_G_I_Dle_clip_3',
    'subset_0002/Send_A_Letter_To_You_Chinese_Dance',
    'subset_0002/Send_A_Letter_To_You_Chinese_Dance_clip_1',
    'subset_0002/Several_Methods_Of_Ballet_Idling',
    'subset_0002/Several_Methods_Of_Ballet_Idling_clip_1',
    'subset_0002/Several_Methods_Of_Ballet_Idling_clip_2',
    'subset_0002/Several_Methods_Of_Ballet_Idling_clip_3',
    'subset_0002/Several_Methods_Of_Ballet_Idling_clip_4',
    'subset_0002/Several_Methods_Of_Ballet_Idling_clip_5',
    'subset_0002/Such_A_Fresh_Melody_Paired_With_A_Gentle_And_Cute_Little_Dance',
    'subset_0002/Summer_Camp_Begins',
    'subset_0002/Summer_Camp_Begins_clip_1',
    'subset_0003/Summer_Camp_Begins',
    'subset_0004/Are_You_Happy_Today',
    'subset_0004/Chasing_The_Light_Become_The_Light',
    'subset_0004/Classical_Dance_Clip1_Clip2_Clip3_Clip4',
    'subset_0004/Classical_Dance_Eye_Training',
    'subset_0004/Classical_Dance_Eye_Training_clip_1',
    'subset_0004/Classical_Dance_Eye_Training_clip_2',
    'subset_0004/Classical_Dance_Mountain_Ghost',
    'subset_0004/Classical_Dance_Mountain_Ghost_clip_1',
    'subset_0004/Dance_Daily_Lovedive',
    'subset_0004/Really_After_Work',
    'subset_0004/Really_After_Work_clip_1',
    'subset_0004/Really_After_Work_clip_2',
    'subset_0004/Really_After_Work_clip_3',
    'subset_0004/Really_After_Work_clip_4',
    'subset_0004/Really_After_Work_clip_5',
    'subset_0004/Really_After_Work_clip_6',
    'subset_0004/Really_After_Work_clip_7',
    'subset_0004/Silk_Fan_Dance',
    'subset_0004/Some_Social_Fear',
    'subset_0004/Some_Social_Fear_clip_1',
    'subset_0004/Super_Dance',
    'subset_0004/Super_Dance_clip_1',
    'subset_0004/Swish_Swish_Is_Just_An_Ordinary_Blacksmith',
    'subset_0004/Swish_Swish_Is_Just_An_Ordinary_Blacksmith_clip_1',
    'subset_0004/To_The_Cloud_Clip1',
    'subset_0004/Wagging_Tail_Hyuna',
    'subset_0004/Wagging_Tail_Hyuna_clip_1',
    'subset_0004/What_Is_A_Person_Full_Of_Desire',
    'subset_0004/What_Is_A_Person_Full_Of_Desire_Clip1',
    'subset_0004/What_Is_A_Person_Full_Of_Desire_Clip1_clip_1',
    'subset_0004/What_Is_A_Person_Full_Of_Desire_clip_1',
    'subset_0004/Which_Pose_Works_Best',
    'subset_0004/Which_Pose_Works_Best_clip_1',
    'subset_0004/Which_Pose_Works_Best_clip_2',
 ]
"""
