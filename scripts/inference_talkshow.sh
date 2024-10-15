# Inference for Avatars with Expression Control
python main.py \
    --stage gs \
    --log.exp_name "w_expr/a_chef_dressed_in_white" \
    --log.eval_only True \
    --prompt.scene demo,talkshow \
    --data.eval_elevation 90 \
    --data.eval_video_fps 30 \
    --data.eval_camera_track fixed \
    --predefined_body_parts hands,face
