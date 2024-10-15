scenes=(
    # "motionx_reenact,kungfu/Aerial_Kick_Kungfu_wushu_14_clip2"
    # "motionx_reenact,kungfu/Aerial_Kick_Kungfu_wushu_21_clip1"
    # "motionx_reenact,kungfu/Shaolin_KungFu_Staff_Workout_Training_3_clip2"
    # "motionx_reenact,kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step15_clip1"
    # "motionx_reenact,kungfu/Yang_style_40_form_Tai_Chi_Competition_routine_step9_clip1"
    # "motionx_reenact,animation/Ways_to_Catch_360_clip1"
    # "motionx_reenact,animation/Ways_to_Catch_Between_the_Legs_clip1"
    # "motionx_reenact,animation/Ways_to_Catch_Large_Ball_clip1"
    # "motionx_reenact,animation/Ways_to_Jump_+_Sit_+_Fall_Broken_Ankle_clip1"
    # "motionx_reenact,animation/Ways_to_Jump_+_Sit_+_Fall_Fun_Photo_clip1"
    # "motionx_reenact,fitness/_BURPEES_clip4"
    # "motionx_reenact,fitness/DIAMOND_LEG_ROTATION_clip5"
    # "motionx_reenact,haa500/baseball_pitch_11_clip1"
    # "motionx_reenact,haa500/basketball_dribble_4_clip1"
    # "motionx_reenact,haa500/basketball_shoot_16_clip1"
    # "motionx_reenact,humman/After_standing_leg_lifts_R_1_clip1"
    # "motionx_reenact,music/Play_the_stringed_guqin_11_clip2"
    # "motionx_reenact,perform/eye_training_clip3"
    "motionx_reenact,perform/peking_opera_performance_man_clip3"
)

for scene in ${scenes[@]}; do
    # Inference for Avatars with Expression Control
    python main.py \
        --stage gs \
        --log.exp_name "w_expr/a_chef_dressed_in_white" \
        --predefined_body_parts hands,face \
        --log.eval_only True \
        --prompt.centralize_pelvis False \
        --prompt.scene "${scene}" \
        --render.use_video_background "${scene}" \
        --data.eval_camera_track predefined
    
    # Inference for Avatars without Expression Control
    python main.py \
        --stage gs \
        --log.exp_name "wo_expr/a_clown" \
        --predefined_body_parts hands \
        --log.eval_only True \
        --prompt.centralize_pelvis False \
        --prompt.scene "${scene}" \
        --render.use_video_background "${scene}" \
        --data.eval_camera_track predefined
done
