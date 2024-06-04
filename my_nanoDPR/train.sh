    # --debug \
# in cuda7, use gpu_ids 0 for 0, 1 for 3
# in cuda8, use gpu_ids 0 for first A6000, 1 for the latter A5000, 好像也不一定@@
    # --multi_gpu \
# accelerate launch --config_file {path/to/config/accelerate_config_file.yaml} \
# {script_name.py} {--arg1} {--arg2} ..
# every 10 min, check if embeddings/hotpot/dpr-multiset/dev_1000_norm.pt exists.
# if so, start the below.

# for dataset in hotpot trivia nq
dataset=nq
accelerate launch \
    --gpu_ids 0 \
    train_dpr.py \
    --config_file "config/dpr_${dataset}.yaml"