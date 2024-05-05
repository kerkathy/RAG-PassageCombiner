# in cuda7, use gpu_ids 0 for 0, 1 for 3
# in cuda8, use gpu_ids 1 for the latter A5000
    # --multi_gpu \
accelerate launch \
    --gpu_ids 1 \
    --debug \
    train_dpr.py 