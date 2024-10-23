# Inference for Avatars without Expression Control using custom motions from TRAM estimations
python main.py \
    --stage gs \
    --log.exp_name "wo_expr/goku" \
    --log.eval_only True \
    --prompt.scene tram,tram \
    --data.eval_video_fps 30 \
    --data.eval_camera_track predefined \
    --predefined_body_parts hands
