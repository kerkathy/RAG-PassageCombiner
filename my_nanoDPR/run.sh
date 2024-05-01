# in cuda7, use gpu_ids 0,1
    # --multi_gpu \
accelerate launch \
    --gpu_ids 0 \
    --debug \
    train_dpr.py 