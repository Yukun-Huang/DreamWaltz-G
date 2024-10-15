import os
import random
import numpy as np
import torch
from typing import Tuple
from configs import PromptConfig


class TextEmbeddings():
    def __init__(self, encode_func) -> None:
        self.encode_func = encode_func

    @torch.inference_mode()
    def init_text_embeddings(self, diffusion) -> Tuple:
        text = self.cfg.guide.text
        negative_text = self.cfg.guide.negative_text
        # Encoding prompts
        text_embeds = diffusion.get_text_embeds(
            prompt=[text],
            negative_prompt=[negative_text],
        )  # [2, 77, 768]
        # Encoding view-dependent prompts
        text_embeds_viewed = []
        if self.cfg.prompt.text_augmentation:
            for viewed_text in self.view_prompt.texts:
                text_embeds_v = diffusion.get_text_embeds(
                    prompt=[viewed_text],
                    negative_prompt=[negative_text],
                )  # [2, 77, 768]
                text_embeds_viewed.append(text_embeds_v)
            # text_embeds_viewed = torch.stack(text_embeds_viewed)  # [N, 2, 77, 768]
        # Return
        return text_embeds, text_embeds_viewed


class TextAugmentation:
    def __init__(self, text: str, cfg: PromptConfig) -> None:
        self.mode = cfg.text_augmentation_mode
        self.azimuth_range, self.elevation_range = self.get_angle_ranges(cfg.angle_front, cfg.angle_overhead)
        self.texts = self.get_view_augmented_texts(text)
        if self.mode in ('dreamwaltz', 'dreamwaltz-g'):
            texts, part2index = self.get_body_part_augmented_texts(text, len(self.texts))
            self.texts.extend(texts)
            self.part2index = part2index
        else:
            self.part2index = None

    def get_angle_ranges(self, angle_front, angle_overhead):
        assert 0 <= angle_front <= 180
        azimuth_range = sorted([
            angle_front/2,
            180 - angle_front/2,
            180 + angle_front/2,
            360 - angle_front/2,
        ])
        assert 0 <= angle_overhead <= 90
        elevation_range = sorted([
            angle_overhead,
            180 - angle_overhead,
        ])
        return azimuth_range, elevation_range

    def get_view_augmented_texts(self, text: str) -> list:
        # Text Augmentation: SJC
        if self.mode == 'prefix':
            texts = [
                f'front view of {text}',
                f'side view of {text}',
                f'backside view of {text}',
                f'side view of {text}',
                f'overhead view of {text}',
                f'bottom view of {text}',
            ]
        # Text Augmentation: Latent-NeRF, DreamFusion
        elif self.mode == 'suffix':
            texts = [
                f'{text}, front view',
                f'{text}, side view',
                f'{text}, back view',
                f'{text}, side view',
                f'{text}, overhead view',
                f'{text}, bottom view',
            ]
        # Text Augmentation: DreamWaltz
        elif self.mode == 'dreamwaltz':
            texts = [
                f'front view of {text}',
                f'side view of {text}',
                f'back view of {text}',
                f'side view of {text}',
                f'overhead view of {text}',
                f'bottom view of {text}',
            ]
        elif self.mode == 'dreamwaltz-g':
            texts = [
                f'front view of {text}',
                f'left side view of {text}',
                f'back view of {text}',
                f'right side view of {text}',
                f'overhead view of {text}',
                f'bottom view of {text}',
            ]
        else:
            raise NotImplementedError(f'{self.mode}')
        # Text
        return texts

    def get_body_part_augmented_texts(self, text: str, start_idx: int):
        texts = [
            f'head of {text}',
            f'face of {text}',
            f'left arm of {text}',
            f'right arm of {text}',
            f'left hand of {text}',
            f'right hand of {text}',
            f'left foot of {text}',
            f'right foot of {text}',
        ]
        parts = ['head', 'face', 'arm_left', 'arm_right', 'hand_left', 'hand_right', 'foot_left', 'foot_right']
        part2index = {}
        for i, part in enumerate(parts):
            part2index[part] = start_idx + i
        return texts, part2index

    def __call__(self, azim, elev, part=None):
        #                    azimuth [B,];      elevation: [B,]
        # front = 0          [0, front)
        # side (left) = 1    [front, 180)
        # back = 2           [180, 180+front)
        # side (right) = 3   [180+front, 360)
        # top = 4                               [0, overhead]
        # bottom = 5                            [180-overhead, 180]

        # init
        elevation = self.elevation_range
        azimuth = self.azimuth_range

        res = torch.zeros(azim.shape[0], dtype=torch.long)

        # first determine by azim
        res[azim >= azimuth[3] or azim < azimuth[0]] = 0
        res[azimuth[0] <= azim < azimuth[1]] = 1
        res[azimuth[1] <= azim < azimuth[2]] = 2
        res[azimuth[2] <= azim < azimuth[3]] = 3

        # then override by elev
        res[elev < elevation[0]] = 4
        res[elev > elevation[1]] = 5

        # for avatar
        # if part is not None and part in self.part2index:
        #     res[...] = self.part2index[part]

        return res
