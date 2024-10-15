age="adult"         # SMPL-X Age: ("adult", "kid")
gender="neutral"    # SMPL-X Gender: ("neutral", "male", "female")
train_res=512       # Training Resolution: If CUDA OOM, reduce this value

exp_name="pretrain_nerf/${age}_${gender}"
python main.py \
    --prompt.smpl_gender ${gender} \
    --prompt.smpl_age ${age} \
    --log.exp_name "${exp_name}" \
    --log.pretrain_only True \
    --stage nerf \
    --optim.fp16 True \
    --optim.iters 5000 \
    --nerf.bg_mode none \
    --guide.controlnet_condition depth_raw \
    --data.train_w ${train_res} \
    --data.train_h ${train_res} \
    --data.body_prob 0.7 \
    --data.face_prob 0.1 \
    --data.hand_prob 0.1 \
    --data.foot_prob 0.1 \
    --data.elevation_range "[30,150]"
