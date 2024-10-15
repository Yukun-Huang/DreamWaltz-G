import pyrallis
import os.path as osp
import numpy as np
from configs import TrainConfig
from configs.prompts import get_avatar_list
from core.trainer import Trainer


def update_path(path, dirname):
    if path is None:
        return None
    if '@' in path:
        path = path.replace('@', dirname)
    return path


def parse_indices(num_prompts, opts):
    if len(opts) == 0:
        prompt_indices = [i for i in range(1, num_prompts+1)]
    else:
        opts = opts[0]
        if '-' in opts:
            start_i, end_i = list(map(int, opts.split('-')))
            prompt_indices = [i for i in range(start_i, end_i + 1)]
        else:
            prompt_indices = eval(opts)
            if isinstance(prompt_indices, int):
                prompt_indices = [prompt_indices]
    return prompt_indices


def run(cfg: TrainConfig):
    trainer = Trainer(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    elif cfg.log.pretrain_only:
        trainer.pretrain()
    elif cfg.log.nerf2gs:
        trainer.pretrain_nerf2gs()
    else:
        trainer.train()


def run_multiple(cfg: TrainConfig):
    set_name, *opts = cfg.guide.text_set.split(',', maxsplit=1)
    prompts = get_avatar_list(set_name)

    prompt_indices = parse_indices(num_prompts=len(prompts), opts=opts)

    assert '@' in cfg.log.exp_name, 'exp_name must contain "@" for inserting text prompts'

    default_cfgs = {
        'exp_name': cfg.log.exp_name,
        'from_nerf': cfg.render.from_nerf,
        'ckpt': cfg.optim.ckpt,
        'ckpt_extra': cfg.optim.ckpt_extra,

        'smpl_age': cfg.prompt.smpl_age,
        'smpl_gender': cfg.prompt.smpl_gender,
    }

    for k in (np.array(prompt_indices) - 1).tolist():
        
        metadata = prompts[k]

        if type(metadata) is str:
            cfg.guide.text = prompts[k]
            cfg.prompt.smpl_age = default_cfgs['smpl_age']
            cfg.prompt.smpl_gender = default_cfgs['smpl_gender']
        elif type(metadata) is dict:
            cfg.guide.text = prompts[k]['text_prompt']
            cfg.prompt.smpl_age = metadata.get('smpl_age', default_cfgs['smpl_age'])
            cfg.prompt.smpl_gender = metadata.get('smpl_gender', default_cfgs['smpl_gender'])

        dirname = '{:04d}_{}'.format(k+1, cfg.guide.text.replace(' ', '_')[:50])
        
        cfg.log.exp_name = update_path(default_cfgs['exp_name'], dirname)
        cfg.render.from_nerf = update_path(default_cfgs['from_nerf'], dirname)
        cfg.optim.ckpt = update_path(default_cfgs['ckpt'], dirname)
        cfg.optim.ckpt_extra = update_path(default_cfgs['ckpt_extra'], dirname)

        try:
            run(cfg)
        except Exception as e:
            print(e)


@pyrallis.wrap()
def main(cfg: TrainConfig):

    if cfg.guide.text_set is None:
        # single prompt
        run(cfg)
    else:
        # multiple prompts
        run_multiple(cfg)


if __name__ == '__main__':
    main()
