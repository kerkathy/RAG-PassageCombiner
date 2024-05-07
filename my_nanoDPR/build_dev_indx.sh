# in cuda7, use gpu_ids 0 for 0, 1 for 3
    # --multi_gpu \
    # --debug \
accelerate launch \
    --gpu_ids 1 \
    build_dev_indx.py