import math
import torch
import bisect
from pathlib import Path
from functools import partial
import numpy as np
from loguru import logger
from typing import Any, Union, Optional, Iterable
from numbers import Number
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from diffusers import SchedulerMixin, DDPMScheduler, DDIMScheduler

from configs import GuideConfig


def C(value: Any, current_step: int, max_iteration: Optional[int] = None) -> float:
    if isinstance(value, Number):
        return value
    else:
        if not isinstance(value, Iterable):
            raise TypeError("Scalar specification only supports Iterable, got", type(value))
        value = list(value)
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if max_iteration is not None and isinstance(start_step, float) and isinstance(end_step, float):
            start_step = int(max_iteration * start_step)
            end_step = int(max_iteration * end_step)
        r = (current_step - start_step) / (end_step - start_step)
        r = max(min(1.0, r), 0.0)
        return start_value + (end_value - start_value) * r


class PriorFunction:
    # DreamTime: Normal Distribution
    # normal_m1: Optional[float] = 800
    # normal_s1: Optional[float] = 300
    # normal_m2: Optional[float] = 500
    # normal_s2: Optional[float] = 100
    WEIGHT_PRIORS = ('uniform', 'normal', 'ddpm', 'p2')

    def __init__(self, weight_prior, annealing_args, t_min: int, t_max: int, scheduler: DDPMScheduler = None, num_train_timesteps=1000) -> None:
        # Init
        self.t_min, self.t_max = t_min, t_max
        self.num_train_timesteps = num_train_timesteps
        self.basic_weight_priors = {
            'uniform': partial(self.uniform_weights,),
            'normal': partial(self.normal_weights, annealing_args=annealing_args),
            'ddpm': partial(self.ddpm_weights, scheduler=scheduler),
            'p2': partial(self.p2_weights, scheduler=scheduler),
        }
        self.weight_priors = list(self.basic_weight_priors.keys()) + ['dreamtime']
        for p in self.basic_weight_priors.keys():
            self.weight_priors.append('dreamtime-' + p)
        # Weight Prior
        if weight_prior.startswith('dreamtime'):
            _, *basic_prior = weight_prior.split('-')
            if len(basic_prior) == 0:
                basic_prior = ('ddpm',)
            weights = self.basic_weight_priors[basic_prior[0]]() * self.basic_weight_priors['normal']()
        else:
            weights = self.basic_weight_priors[weight_prior]()
        # Limitation
        weights = weights[t_min : t_max + 1]
        weights = weights / torch.sum(weights)
        # Convert Weights to Iteration-Time Mapping
        weights_flip = weights.flip(dims=(0,))
        weights_cumsum = (weights_flip).cumsum(dim=0).detach().cpu().numpy()
        # Setup
        self.weights = weights
        self.weights_cumsum = weights_cumsum

    def uniform_weights(self):
        # weights
        w = []
        for _ in range(self.num_train_timesteps):
            w.append(1.0)
        return torch.tensor(w)

    def normal_weights(self, annealing_args):
        # config
        if len(annealing_args) == 2:
            m1, s1 = float(annealing_args[0]), float(annealing_args[1])
            m2, s2 = m1, s1
        elif len(annealing_args) == 4:
            m1, s1, m2, s2 = float(annealing_args[0]), float(annealing_args[1]), float(annealing_args[2]), float(annealing_args[3])
        assert m1 >= m2
        # weights
        w = []
        for t in range(self.num_train_timesteps):
            if t > m1:
                w.append(math.exp(- (t - m1)**2 / (2 * s1**2)))
            elif m2 <= t <= m1:
                w.append(1.0)
            elif t < m2:
                w.append(math.exp(- (t - m2)**2 / (2 * s2**2)))
            else:
                assert 0, f'{t}, {m1}, {m2}'
        return torch.tensor(w)

    def ddpm_weights(self, scheduler: DDPMScheduler):
        base_weights = ((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5
        return base_weights

    def p2_weights(self, scheduler: DDPMScheduler, p2_k=1.0, p2_gamma=1.0, base_weights=None):
        # "Perception Prioritized Training of Diffusion Models", CVPR 2022.
        snr = 1.0 / (1 - scheduler.alphas_cumprod) - 1
        if base_weights is None:
            base_weights = (1 - scheduler.betas) * (1 - scheduler.alphas_cumprod) / scheduler.betas
        p2_weights = base_weights / (p2_k + snr) ** p2_gamma
        return p2_weights

    def __call__(self, train_step, max_iteration):
        delta_timestep = bisect.bisect_left(self.weights_cumsum, train_step / max_iteration)
        return max(self.t_max - delta_timestep, self.t_min)


class WindowedAnnealing:
    def __init__(self, cfg: GuideConfig, t_min: int, t_max: int, scheduler: DDPMScheduler = None) -> None:
        # Init
        self.annealing_type, *annealing_args = cfg.time_annealing.split(',')
        self.window_type, *window_args = cfg.time_annealing_window.split(',')
        # Time Annealing Window Config
        window_size = None
        if len(window_args) == 2:
            window_direction = window_args[0]
            window_size = int(window_args[1])
        elif len(window_args) == 1:
            window_direction = window_args[0]
        elif len(window_args) == 0:
            window_direction = 'middle'
        else:
            assert 0, f'Invalid annealing window args: {window_args}'
        self.window_size = window_size
        # Setup
        self.annealing_function = self.build_annealing_function(annealing_args, t_min, t_max, scheduler)
        self.window_function = self.build_window_function(window_direction, t_min, t_max)

    def build_window_function(self, window_direction, t_min, t_max):
        if window_direction == 'tail':
            assert self.window_size is not None
        adaptive = self.window_size is None
        if self.window_type == 'impluse':
            def window_function(t):
                return t
        elif self.window_type == 'square':
            def window_function(t):
                if window_direction == 'lower':
                    t_lower = t_min if adaptive else max(t_min, t - self.window_size)
                    t = np.random.randint(t_lower, t + 1)  # [low, high)
                elif window_direction == 'upper':
                    t_upper = t_max if adaptive else min(t_max, t + self.window_size)
                    t = np.random.randint(t, t_upper + 1)  # [low, high)
                elif window_direction == 'middle':
                    if adaptive:
                        window_size = min(t_max - t, t - t_min)
                        t = np.random.randint(t - window_size,
                                              t + window_size + 1)
                    else:
                        t = np.random.randint(max(t_min, t - self.window_size // 2),
                                              min(t_max, t + self.window_size // 2) + 1)
                elif window_direction == 'tail':
                    t_upper = t_min + self.window_size
                    if t < t_upper:
                        t = np.random.randint(t_min, t_upper + 1)
                else:
                    assert 0, f'Invalid window direction: {window_direction}'
                return t
        elif self.window_type == 'normal':
            def window_function(t):
                if window_direction == 'lower':
                    t_mean = (t_min + t) / 2 if adaptive else t - self.window_size / 2
                    sigma = (t - t_min) / 6
                elif window_direction == 'upper':
                    t_mean = (t_max + t) / 2 if adaptive else t + self.window_size / 2
                    sigma = (t_max - t) / 6
                elif window_direction == 'middle':
                    t_mean = t
                    sigma = min(t_max - t, t - t_min) / 6
                elif window_direction == 'tail':
                    if t >= self.window_size:
                        t_mean, sigma = t, 0.0
                    else:
                        t_upper = t_min + self.window_size
                        t_mean = (t_min + t_upper) / 2
                        sigma = (t_upper - t_min) / 6
                else:
                    assert 0, f'Invalid window direction: {window_direction}'
                if not adaptive:
                    sigma = self.window_size / 6
                t = -1
                while t < t_min or t > t_max:
                    t = int(np.random.normal(t_mean, sigma))
                return t
        else:
            assert 0, f'Invalid time annealing window type: {self.window_type}'
        # Return
        logger.info(f'[TimestepScheduler] window_function: {self.window_type}, direction: {window_direction}, window_size: {self.window_size}')
        return window_function

    def build_annealing_function(self, annealing_args, t_min, t_max, scheduler):
        # Annealing Function with Prior
        if self.annealing_type in PriorFunction.WEIGHT_PRIORS or self.annealing_type.startswith('dreamtime'):
            annealing_function = PriorFunction(self.annealing_type, annealing_args, t_min=t_min, t_max=t_max, scheduler=scheduler)
            t_begin, t_end = t_max, t_min
        # Annealing Function without Prior
        else:
            # Time Annealing Config
            if self.annealing_type == 'linear':
                p = 1
            elif self.annealing_type == 'hifa':
                p = 0.5
            else:
                p = None
            if len(annealing_args) == 3:
                t_begin, t_end, p = int(annealing_args[0]), int(annealing_args[1]), float(annealing_args[2])
            elif len(annealing_args) == 2:
                t_begin, t_end = int(annealing_args[0]), int(annealing_args[1])
            elif len(annealing_args) == 0:
                t_begin, t_end = t_max, t_min
                logger.warning(f'Please manually specify the annealing range! Now using [{t_begin}, {t_end}]')
            else:
                assert 0, f'Invalid annealing args: {annealing_args}'
            assert t_begin >= t_end and t_max >= t_begin and t_min <= t_end
            # Annealing Function
            def annealing_function(i, max_iter):
                return int(t_begin - (t_begin - t_end) * (i / max_iter) ** p)
        # Return
        logger.info(f'[TimestepScheduler] annealing_function: {self.annealing_type}, [t_min, t_max]: [{t_min}, {t_max}], vary: {t_begin}->{t_end}')
        return annealing_function

    def __call__(self, train_step, max_iteration, use_window=True) -> int:
        t = self.annealing_function(train_step, max_iteration)
        if use_window:
            t = self.window_function(t)
        return t


class TimePrioritizedScheduler:
    def __init__(self,
            cfg: GuideConfig,
            device,
            scheduler: Union[str, SchedulerMixin] = 'ddpm',
            total_timesteps=1000,
            num_train_timesteps=1000,
            pretrained_model_name_or_path=None,
        ) -> None:

        # Scheduler
        if scheduler == 'ddpm':
            self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        elif scheduler == 'ddim':
            self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        else:
            self.scheduler: SchedulerMixin = scheduler
        self.add_noise = self.scheduler.add_noise

        # Hyper Params
        self.device = device
        self.time_sampling = cfg.time_sampling
        self.min_step_cfg = cfg.min_timestep
        self.max_step_cfg = cfg.max_timestep
        self.num_train_timesteps = num_train_timesteps
        self.total_timesteps = total_timesteps

        # Placeholder
        self.train_step: int = None
        self.max_iteration: int = None

        if total_timesteps < num_train_timesteps:
            self.scheduler.set_timesteps(total_timesteps)
            self.downsampled_timesteps = self.scheduler.timesteps.flip(dims=(0,)).to(device)  # 21, 41, ..., 961, 981

        # Coefficients
        self.betas = self.scheduler.betas.to(self.device)
        self.alphas = self.scheduler.alphas.to(self.device)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = torch.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)

        # Timestep Mode
        self.time_annealing = None
        if self.time_sampling == 'annealed':
            self.time_annealing = WindowedAnnealing(cfg, t_min=self.min_step, t_max=self.max_step, scheduler=self.scheduler)
        elif self.time_sampling.startswith('stage'):
            self.time_sampling, *time_sampling_args = self.time_sampling.split('-')
            self.num_stage = int(time_sampling_args[0]) if len(time_sampling_args) else 2
            self.stage_intervals = self.get_stage_division(self.num_stage)
        
    @property
    def min_step(self):
        min_step = C(self.min_step_cfg, current_step=self.train_step, max_iteration=self.max_iteration)
        return int(self.num_train_timesteps * min_step)

    @property
    def max_step(self):
        max_step = C(self.max_step_cfg, current_step=self.train_step, max_iteration=self.max_iteration)
        return int(self.num_train_timesteps * max_step)

    def set_train_state(self, train_step, max_iteration):
        self.train_step = train_step
        self.max_iteration = max_iteration

    def get_stage_division(self, num_stage):
        timesteps_per_stage = (self.max_step - self.min_step) // num_stage
        stage_intervals = []
        for i in range(num_stage, 0, -1):
            stage_intervals.append(
                (self.min_step + timesteps_per_stage*(i-1), self.min_step + timesteps_per_stage*i),
            )
        print(stage_intervals)
        return stage_intervals

    def quantify_timestep(self, timestep: torch.Tensor):
        if self.total_timesteps < self.num_train_timesteps:
            timestep_index = (timestep * self.total_timesteps / self.num_train_timesteps).long()
            timestep = self.downsampled_timesteps[timestep_index]
        return timestep

    def get_timestep(self, batch_size, train_step, max_iteration):
        
        self.set_train_state(train_step=train_step, max_iteration=max_iteration)

        if self.time_sampling == 'uniform':
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, [batch_size,], dtype=torch.long, device=self.device)
        
        elif self.time_sampling == 'constant':
            mid_step = (self.min_step + self.max_step) // 2
            t = torch.randint(mid_step, mid_step + 1, [batch_size,], dtype=torch.long, device=self.device)
        
        elif self.time_sampling == 'linear':
            timestep_delta = (self.max_step - self.min_step) / (max_iteration - 1)
            timestep = int(self.max_step - (train_step - 1) * timestep_delta)
            t = torch.ones([batch_size,], dtype=torch.long, device=self.device) * timestep
        
        elif self.time_sampling == 'stage':
            iters_per_stage = max_iteration // self.num_stage
            i_stage = min(train_step // iters_per_stage, self.num_stage - 1)
            min_step, max_step = self.stage_intervals[i_stage]
            min_step = self.min_step  # Important!
            t = torch.randint(min_step, max_step + 1, [batch_size,], dtype=torch.long, device=self.device)
        
        elif self.time_sampling == 'annealed':
            timestep = self.time_annealing(train_step, max_iteration, use_window=False)
            t = torch.ones([batch_size,], dtype=torch.long, device=self.device) * timestep
        
        else:
            raise NotImplementedError
    
        return self.quantify_timestep(t)

    def draw_curves(self, save_dir: Union[Path, str]):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        save_dir.mkdir(exist_ok=True)

        def _draw(dat_x, dat_y, label):
            if isinstance(dat_x, torch.Tensor):
                dat_x = dat_x.detach().cpu().numpy()
            if isinstance(dat_y, torch.Tensor):
                dat_y = dat_y.detach().cpu().numpy()
            plt.plot(dat_x, dat_y, label=label)

        def _create_and_draw(fig_name, dat_x, dat_y, label=None, limit=True):
            plt.figure()
            _draw(dat_x, dat_y, label)
            if limit:
                plt.xlim([0.0, 1.0])
                plt.ylim([0, self.num_train_timesteps])
            if label is not None:
                plt.legend()
            plt.savefig(save_dir / fig_name)
            plt.clf()

        # Iteration-to-Timestep
        train_steps = list(range(1, 1000+1))
        i_each_step = [i / len(train_steps) for i in train_steps]
        t_each_step = [self.get_timestep(1, i, len(train_steps)).item() for i in train_steps]
        _create_and_draw(fig_name='iter_to_t.png', dat_x=i_each_step, dat_y=t_each_step)
        if self.time_annealing is not None:
            t_each_step = [self.get_timestep(1, i, len(train_steps)).item() for i in train_steps]
            _create_and_draw(fig_name='iter_to_t_boundary.png', dat_x=i_each_step, dat_y=t_each_step)
            if isinstance(self.time_annealing.annealing_function, PriorFunction):
                t_min = self.time_annealing.annealing_function.t_min
                t_max = self.time_annealing.annealing_function.t_max
                t_vals = train_steps[t_min : t_max + 1]
                t_weights = self.time_annealing.annealing_function.weights
                _create_and_draw(fig_name='t_weights.png', dat_x=t_vals, dat_y=t_weights, limit=False)


class TimePrioritizedLR:
    def __init__(self, optimizer, tp_scheduler: TimePrioritizedScheduler) -> None:
        prior_function_args = {
            't_min': 0,
            't_max': 999,
            'scheduler': tp_scheduler.scheduler,
        }
        self.prior_function = PriorFunction(weight_prior='ddpm', annealing_args=None, **prior_function_args)
        weights = self.prior_function.weights
        self.weights = weights / torch.max(weights)
        self.current_timestep = None
        self.current_weight = None
        # record learning rate
        self.default_lr = optimizer.defaults['lr']
        self.param_lrs = []
        for params in optimizer.param_groups:
            if 'lr' in params:
                self.param_lrs.append(params['lr'])
            else:
                self.param_lrs.append(None)

    def set_learning_rate(self, timestep, optimizer: Optimizer):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
        if timestep != self.current_timestep:
            self.current_timestep = timestep
            self.current_weight = self.get_learning_rate_weight(timestep)
            for params, lr in zip(optimizer.param_groups, self.param_lrs):
                if lr is not None:
                    params['lr'] = lr * self.current_weight
                else:
                    params['lr'] = self.default_lr * self.current_weight
        return self.current_weight

    def get_learning_rate_weight(self, timestep):
        lr_weight = 1.0
        lr_weight = self.weights[timestep]
        return lr_weight
