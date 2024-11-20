# Inference for Avatars without Expression Control using custom motions from TRAM estimations
python main.py \
    --stage gs \
    --log.exp_name "wo_expr/goku" \
    --log.eval_only True \
    --prompt.scene tram,example_video \
    --prompt.centralize_pelvis False \
    --render.use_video_background "datasets/tram/example_video/inpainted_video.mp4" \
    --data.eval_video_fps 30 \
    --data.eval_camera_track predefined \
    --predefined_body_parts hands
