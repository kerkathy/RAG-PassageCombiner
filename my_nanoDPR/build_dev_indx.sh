# in cuda7, use gpu_ids 0 for 0, 1 for 3
    # --multi_gpu \
    # --debug \
accelerate launch \
    --gpu_ids 0 \
    make_dev_index.py