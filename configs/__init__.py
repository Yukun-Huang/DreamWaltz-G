from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, Any
from loguru import logger
from datetime import datetime


@dataclass
class NeRFConfig:
    """ Parameters for the NeRF Renderer """
    # feature encoding of instant-ngp
    desired_resolution: int = 2048
    num_levels: int = 16
    level_dim: int = 2
    base_resolution: int = 16
    density_activation: str = 'exp'

    # Whether to use CUDA raymarching
    cuda_ray: bool = True
    grid_size: int = 128
    # Maximal number of steps sampled per ray with cuda_ray
    max_steps: int = 1024
    # Number of steps sampled when rendering without cuda_ray
    num_steps: int = 128
    # Number of upsampled steps per ray
    upsample_steps: int = 0
    # Iterations between updates of extra status
    update_extra_interval: int = 16
    # batch size of rays at inference
    max_ray_batch: int = 4096
    # threshold for density grid to be occupied
    density_thresh: float = 10

    # Assume the scene is bounded in box(-bound,bound)
    bound: float = 2
    # dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)
    dt_gamma: float = 0
    # minimum near distance for camera
    min_near: float = 0.1

    # Which NeRF backbone to use
    backbone: str = 'tiledgrid'  # 'tiledgrid'
    # Define the nerf output type
    nerf_type: str = 'rgb'  # latent, rgb, latent_tune
    # Structure
    structure: str = 'shared_mlp'  # shared_mlp, dual_mlp, dual_enc
    # density blob
    density_prior: str = 'none'
    # background augmentation
    bg_mode: str = 'gray'  # {'nerf', 'random', 'white', 'black',  'gray', 'gaussian', 'zero'}
    # if positive, use a background model at sphere(bg_radius)
    bg_radius: float = 3.0
    # random augmentation of background
    rand_bg_prob: Optional[float] = None
    bg_suppress: bool = False
    bg_suppress_dist: float = 0.5
    detach_bg_weights_sum: bool = False

    # DMTet
    dmtet: bool = False
    dmtet_reso_scale: int = 8
    lock_geo: bool = False
    tet_grid_size: int = 128

    # DMTet Loss
    lambda_normal: float = 0.0
    lambda_2d_normal_smooth: float = 0.0
    lambda_3d_normal_smooth: float = 0.0
    lambda_mesh_normal: float = 0.5
    lambda_mesh_laplacian: float = 0.5

    # Optimizer
    optimizer: str = 'adam'
    # Learning rate
    lr: float = 1e-3
    bg_lr: float = 1e-3
    # Start shading at this iteration
    start_shading_iter: Optional[int] = None
    # LR Policy
    lr_policy: str = 'constant'

    # Sparsity Constraint
    lambda_opacity: float = 0.0  # default = [1e-3, 5e-3]
    lambda_entropy: float = 0.0
    lambda_emptiness: float = 0.0  # default = 1.0

    sparsity_multiplier: float = 20
    sparsity_step: float = 1.0

    # Shape Constraint
    lambda_shape: float = 5e-6


@dataclass
class RenderConfig:
    gs_type: str = 'dreamwaltz-g'

    # Deformation Configs
    deform_type: str = 'glbs'
    deform_with_shape: bool = False
    deform_rotation_mode: str = 'quaternion'  # {'none', 'matrix', 'quaternion'}
    
    lbs_lr: float = 1e-4
    betas_lr: float = 1e-2
    deform_learn_v_template: bool = False
    deform_learn_shapedirs: bool = False
    deform_learn_posedirs: bool = False
    deform_learn_expr_dirs: bool = False
    deform_learn_lbs_weights: bool = False
    deform_learn_J_regressor: bool = False

    always_animate: bool = True
    lbs_weight_smooth: bool = False
    lbs_weight_smooth_K: Optional[int] = 30
    lbs_weight_smooth_N: Optional[int] = 5000

    use_joint_shape_offsets: bool = False
    use_vertex_shape_offsets: bool = False
    use_vertex_pose_offsets: bool = False

    use_non_rigid_offsets: bool = True
    use_non_rigid_scales: bool = True
    use_non_rigid_rotations: bool = False
    
    non_rigid_scale_mode: str = 'add'
    non_rigid_rotation_mode: str = 'add'

    # 3D Gaussian Configs
    sh_levels: int = 4

    spatial_scale: Optional[float] = None  # 1.0

    init_opacity: float = 0.99
    init_offset: float = 0.01
    init_scale: float = 0.001
    init_scale_radius_rate: float = 1.0
    max_scale: float = 0.01

    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_mlp_background: bool = False
    use_video_background: Optional[str] = None
    use_gs_background: Optional[str] = None

    gaussian_color_init: str = 'rand'
    gaussian_point_init: str = 'mesh_surface'
    gaussian_scale_init: str = 'default'

    n_gaussians: int = 1000000  # 100000
    n_gaussians_per_vertex: int = 1
    n_gaussians_per_triangle: int = 6

    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    feature_lr: float = 0.0125
    opacity_lr: float = 0.01
    scaling_lr: float = 0.0025
    rotation_lr: float = 0.001

    use_densifier: bool = False

    densify_from_iter: Optional[int] = None
    densify_until_iter: Optional[int] = None

    densify_grad_threshold: float = 100 # 0.0002 for MSE, 100 for SDS

    densify_disable_clone: bool = False
    densify_disable_split: bool = False
    densify_disable_prune: bool = False
    densify_disable_reset: bool = True

    enable_grad_prune: bool = False

    from_nerf: Optional[str] = None
    nerf_resolution: int = 400  # 256
    nerf_exclusion_bboxes: Optional[str] = None

    reset_nerf: bool = False
    
    use_nerf_opacities: bool = True
    use_nerf_scales_and_quaternions: bool = True
    use_nerf_scales: bool = False
    use_nerf_quaternions: bool = False

    use_nerf_encoded_position: bool = True

    use_deform_scales_and_quaternions: bool = False

    use_nerf_mesh_opacities: bool = False
    use_nerf_mesh_scales_and_quaternions: bool = True

    prune_points_close_to_mesh: bool = True
    prune_dists_close_to_mesh: Optional[float] = 0.01

    learn_positions: bool = True
    learn_scales: bool = True
    learn_quaternions: bool = True
    learn_lbs_weights: bool = False

    learn_hand_betas: bool = False
    learn_face_betas: bool = False

    learn_mesh_bary_coords: bool = True
    learn_mesh_vertex_coords: bool = False
    learn_mesh_scales: bool = True
    learn_mesh_quaternions: bool = False

    lambda_outfit_offset: float = 20.0
    lambda_outfit_scale: float = 1.0

    render_mesh_binding_3d_gaussians_only: bool = False
    render_unconstrained_3d_gaussians_only: bool = False
    use_zero_scales: bool = False
    use_constant_colors: Optional[Tuple[float, float, float]] = None
    use_constant_opacities: Optional[float] = None

    avatar_scale: Optional[str] = None
    avatar_transl: Optional[str] = None

    use_fixed_n_gaussians: Optional[int] = None


@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Text
    text: str = ""
    text_set: Optional[str] = None  # e.g., 'DreamFusion,1-10'
    null_text: str = ""
    negative_text: str = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad hands, missing fingers, error, cropped, normal quality, fewer digit, owres, extra digit, worst quality, jpeg artifacts, lowres, bad feet, disfigured, missing arms, long neck, ugly, bad proportions, multiple breasts, fused fingers, extra legs, poorly drawn hands, cloned face, malformed hands, mutated hands and fingers, missing limb, malformed mutated, unclear eyes, fused hand, disappearing thigh, disappearing calf, bad body, on hand with less than 5 fingers, crown, stacked torses, stacked hands, totem pole"
    use_negative_text: bool = True
    
    # "Rethinking Score Distillation as a Bridge Between Image Distributions." Arxiv 2024.
    negative_text_in_SBP: str = "oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed"
    
    # General
    dtype: str = 'fp32'
    grad_viz: bool = False

    # Diffusion Model
    diffusion: str = "sd15"
    diffusion_fp16: bool = False

    # ControlNet
    use_controlnet: bool = True
    controlnet: str = "sd15"
    controlnet_fp16: bool = False
    controlnet_condition: str = 'pose'  # 'pose', 'depth', 'depth,pose', ...
    controlnet_scale: float = 1.0

    # Extra Module
    lora_name: Optional[str] = None  # LoRA model
    concept_name: Optional[str] = None  # Textual-Inversion concept

    # CFG
    guidance_scale: float = 50.0
    guidance_adjust: str = 'constant'

    # Timestep
    min_timestep: Any = 0.020  # 0.020  0.020
    max_timestep: Any = 0.980  # 0.980  0.999
    time_sampling: str = 'uniform' # ['uniform', 'linear', 'step', 'annealed']
    time_annealing: str = 'linear'  # ('linear', 'hifa', 'legacy_dreamtime')
    time_annealing_window: str = 'impluse'  # ('impluse', 'square', 'normal')

    # SDS
    sds_loss_type: str = "sds"
    sds_weight_type: str = "sjc"
    input_interpolate: bool = True

    # Gradients
    grad_latent_clip: bool = False
    grad_latent_clip_scale: float = 3.0
    grad_latent_norm: bool = False
    grad_latent_nan_to_num: bool = False

    grad_rgb_clip: bool = False
    grad_rgb_clip_mask_guidance: bool = False
    grad_rgb_clip_scale: float = 3.0
    grad_rgb_norm: bool = False

    # PGC: Enhancing High-Resolution 3D Generation through Pixel-wise Gradient Clipping
    pgc_clip_rgb: float = -1  # 0.1
    pgc_suppress_type: int = 0
    # SDS Loss, default: 1.0
    lambda_guidance: float = 1.0

    def __post_init__(self):
        self.min_timestep = eval(str(self.min_timestep))
        self.max_timestep = eval(str(self.max_timestep))
        self.controlnet_condition = self.controlnet_condition.split(',')
        self.use_sdxl = self.diffusion.startswith('sdxl')
        if self.use_sdxl and not self.diffusion_fp16:
            logger.warning('NOTICE! Always use fp16 for SDXL model.')
            self.diffusion_fp16 = True


@dataclass
class DataConfig:
    # Render width and height for training
    train_w: Union[int, str] = 512
    train_h: Union[int, str] = 512
    grid_milestone: Optional[str] = None  # "[0.0,0.3,0.7]"
    progressive_grid: bool = True
    # Render width for inference
    eval_w: int = 512
    eval_h: int = 512
    # Render height for inference
    test_w: int = 1024
    test_h: int = 1024
    elevation_range: str = '(60, 120)'
    azimuth_range: str = '(0, 360)'
    fovy_range: Tuple[float, float] = (40, 70)
    radius_range: Tuple[float, float] = (1.0, 2.0)
    z_near: float = 0.01
    z_far: float = 1000.
    progressive_radius: bool = False
    progressive_radius_ranges: Optional[str] = None  # "(2.5,3.5),(1.0,2.0)"

    batched_view: bool = False
    uniform_sphere_rate: float = 0.0

    jitter_pose: bool = False
    vertical_jitter: Optional[Tuple[float, float]] = None  # (-0.5, +0.5)
    use_human_vertical_jitter: bool = True
    camera_offset: Optional[Tuple[float, float, float]] = None

    # Number of angles to sample for eval during training
    eval_size: int = 8
    # Number of angles to sample for eval after training
    full_eval_size: int = 60
    # Render angle elevation for inference
    eval_azimuth: float = 0.0
    eval_elevation: float = 80.0
    # Render radius rate
    eval_radius: Optional[float] = 2.4  # 2.1 ~ 2.4
    eval_radius_rate: float = 1.2
    # Eval options
    eval_save_video: bool = True
    eval_save_image: bool = True
    eval_video_fps: int = 30
    eval_fix_animation: bool = False
    eval_camera_track: str = 'circle'  # ('fixed', 'circle')
    eval_camera_offset: Optional[Tuple[float, float, float]] = None
    eval_bg_mode: Optional[str] = None  # ('gray', 'white', 'black')
    eval_body_part: Optional[str] = None  # (None, 'head', 'left_hand', 'right_hand')

    # Random camera focus for training
    body_prob: float = 0.8
    head_prob: float = 0.0
    face_prob: float = 0.2
    hand_prob: float = 0.0
    arm_prob: float = 0.0
    foot_prob: float = 0.0

    head_azimuth_range: str = '(0, 360)'
    head_elevation_range: str = '(75, 105)'
    head_radius_range: Tuple[float, float] = (0.5, 1.5)

    face_azimuth_range: str = '(0, 90),(270,360)'
    face_elevation_range: str = '(75, 105)'
    face_radius_range: Tuple[float, float] = (0.5, 1.0)

    hand_left_azimuth_range: str = '(0, 180)'
    hand_right_azimuth_range: str = '(180, 360)'
    hand_elevation_range: str = '(60, 120)'
    hand_radius_range: Tuple[float, float] = (0.5, 1.0)

    foot_left_azimuth_range: str = '(0, 360)'
    foot_right_azimuth_range: str = '(0, 360)'
    foot_elevation_range: str = '(75, 105)'
    foot_radius_range: Tuple[float, float] = (0.5, 1.5)

    cameras: Optional[str] = None

    random_pose_iter: int = 0

    # Objaverse
    objaverse_id: str = 'ff30e709302d47a683b5b0e98148b5a7'

    def __post_init__(self):
        self.azimuth_range = eval(self.azimuth_range)
        self.elevation_range = eval(self.elevation_range)

        self.head_azimuth_range = eval(self.head_azimuth_range)
        self.head_elevation_range = eval(self.head_elevation_range)

        self.face_azimuth_range = eval(self.face_azimuth_range)
        self.face_elevation_range = eval(self.face_elevation_range)

        self.hand_left_azimuth_range = eval(self.hand_left_azimuth_range)
        self.hand_right_azimuth_range = eval(self.hand_right_azimuth_range)
        self.hand_elevation_range = eval(self.hand_elevation_range)

        self.foot_left_azimuth_range = eval(self.foot_left_azimuth_range)
        self.foot_right_azimuth_range = eval(self.foot_right_azimuth_range)
        self.foot_elevation_range = eval(self.foot_elevation_range)

        if self.grid_milestone is not None:
            self.grid_milestone = eval(self.grid_milestone)


@dataclass
class PromptConfig:
    # View-dependent Text Augmentation
    text_augmentation: bool = True
    text_augmentation_mode: str = 'dreamwaltz-g'
    # set [-angle_front/2, +angle_front/2] as the front region
    angle_front: float = 90
    # set [0, +angle_overhead] as the overhead region
    angle_overhead: float = 60
    # SMPL Model
    flat_hand_mean: bool = False
    # SMPL Prompt
    smpl_type: str = 'smplx'
    smpl_gender: str = "neutral"
    smpl_age: str = "adult"
    use_smplx_2020_neutral: bool = True
    num_person: Optional[int] = None
    scene: str = 'canonical'  # 'canonical', 'vposer', 'dance', 'basketball', ...
    canonical_pose: str = 'canonical-A-adjust'  # 'canonical', 'canonical-A', 'canonical-T'
    canonical_mixup_prob: float = 0.5
    # SMPL Sequence
    frame_interval: Optional[int] = None
    # SMPL Shape
    canonical_betas: Optional[str] = None
    observed_betas: Optional[str] = None
    pop_betas: bool = True
    max_beta_iteration: int = 25  # for shape interpolation
    # NeRF Depth
    nerf_depth: bool = False
    nerf_depth_step: float = 0.2
    # Others
    centralize_pelvis: bool = True
    pop_transl: bool = False
    normalize_transl: bool = False
    pop_global_orient: bool = False
    # Object
    num_object: int = 0

    # Skeleton Map
    use_occlusion_culling: bool = True
    draw_body_keypoints: bool = True
    draw_hand_keypoints: bool = True
    draw_face_landmarks: bool = False
    ignore_body_self_occlusion: bool = True
    openpose_left_right_flip: bool = False

    adaptive_hand_dist_thres: Optional[float] = None


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Batch size
    batch_size: int = 1

    # Seed for experiment
    seed: int = 0
    # Total iters
    iters: int = 5000  # 10000
    # use amp mixed precision training
    fp16: bool = False
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None
    ckpt_extra: Optional[str] = None


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str = 'default'
    # Experiment output dir
    exp_root: Path = Path('outputs/')
    # How many steps between save step
    save_interval: int = 5000
    # How many steps between snapshot step
    snapshot_interval: int = 500
    evaluate_interval: int = 500
    # Run only test
    eval_only: bool = False
    eval_dirname: Optional[str] = None
    # resume pretrain
    resume_pretrain: bool = True
    # Run only pretrain
    pretrain_only: bool = False
    nvstrain_only: bool = False
    anytrain_only: bool = False
    nerf2gs: bool = False
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 1
    # Skip decoding and vis only depth and normals
    skip_rgb: bool = False
    # Debug mode
    debug: bool = False
    check: bool = False
    check_sd: bool = False

    @property
    def exp_dir(self) -> Path:
        exp_dir = self.exp_root / self.exp_name
        if exp_dir.exists() and not self.eval_only and self.debug:
            exp_dir = Path(str(exp_dir) + datetime.now().strftime("@%Y%m%d-%H%M%S"))
        return exp_dir


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    nerf: NeRFConfig = field(default_factory=NeRFConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    device: str = 'cuda'
    character: Optional[str] = None

    use_sigma_guidance: bool = False
    use_sigma_hand_guidance: bool = False
    use_sigma_face_guidance: bool = False
    sigma_loss_type: str = 'margin'
    sigma_prob: float = 1.0
    sigma_num_points: int = 5000
    
    sigma_surface_thickness: float = 0.005
    sigma_guidance_peak: float = 15.
    sigma_noise_range: float = 0.05
    sigma_guidance_delta: float = 0.2

    lambda_sigma_sigma: float = 1.0
    lambda_sigma_albedo: float = 0.0
    lambda_sigma_normal: float = 0.0

    predefined_body_parts: str = "hands"
    
    stage: str = 'gs'  # 'nerf' or 'gs'

    def __post_init__(self):
        if self.log.eval_only and not self.optim.resume and self.optim.ckpt is None:
            logger.warning('NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True
        if self.log.pretrain_only and self.guide.controlnet_condition[0] != 'depth_raw':
            logger.warning(f'NOTICE! log.pretrain_only=True, but guide.controlnet_condition = {self.guide.controlnet_condition} -> Manually setting guide.controlnet_condition to depth_raw')
            self.guide.controlnet_condition = ['depth_raw']

        if self.log.nerf2gs:
            if self.stage != 'gs':
                logger.warning('NOTICE! log.nerf2gs = True, but stage != "gs". Manually setting stage to "gs".')
                self.stage = 'gs'
            assert self.render.from_nerf is not None
