    # --debug \
# in cuda7, use gpu_ids 0 for 0, 1 for 3
# in cuda8, use gpu_ids 0 for first A6000, 1 for the latter A5000, 好像也不一定@@
    # --multi_gpu \
# accelerate launch --config_file {path/to/config/accelerate_config_file.yaml} \
# {script_name.py} {--arg1} {--arg2} ...
# for i in 1 2
# do
#     accelerate launch \
#         --gpu_ids 2 \
#         train_dpr.py \
#         --config_file "config/24G_train_dpr_nq.yaml"
# done
accelerate launch \
    --gpu_ids 2 \
    train_dpr.py \
    --config_file "config/train_dpr_hotpot.yaml"