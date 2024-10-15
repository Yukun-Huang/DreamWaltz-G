# Inference for Avatars with Expression Control
python main.py \
    --stage gs \
    --log.exp_name "w_expr/a_chef_dressed_in_white" \
    --log.eval_only True \
    --prompt.scene canonical \
    --predefined_body_parts hands,face

# Inference for Avatars without Expression Control
python main.py \
    --stage gs \
    --log.exp_name "wo_expr/a_clown" \
    --log.eval_only True \
    --prompt.scene canonical \
    --predefined_body_parts hands
