# in cuda7, use gpu_ids 0 for 0, 1 for 3
# in cuda8, use gpu_ids 0 for first A6000, 1 for the latter A5000, 好像也不一定@@
    # --multi_gpu \
dataset=nq
accelerate launch \
    --gpu_ids 2 \
    only_eval.py \
    --config_file "config/eval_dpr_${dataset}.yaml"