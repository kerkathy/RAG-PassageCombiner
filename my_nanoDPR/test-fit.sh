    # --debug \
# in cuda7, use gpu_ids 0 for 0, 1 for 3
# in cuda8, use gpu_ids 0 for first A6000, 1 for the latter A5000, 好像也不一定@@
    # --multi_gpu \
# accelerate launch --config_file {path/to/config/accelerate_config_file.yaml} \
# {script_name.py} {--arg1} {--arg2} ...
for i in {6..9}
do
    echo "Run $i th sweep"
    accelerate launch \
        --gpu_ids 0 \
        train_dpr_test-fit.py \
        --config_file "config/debug-24G_train_dr_nq copy $i.yaml"
    echo "Done $ith sweep"
done