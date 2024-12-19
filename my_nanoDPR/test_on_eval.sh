for dataset in hotpot trivia nq
do
    config_file="config/test_dpr_${dataset}_eval.yaml"
    echo "Running $config_file"

    if [ ! -f $config_file ]; then
        echo "$config_file does not exist. Skipping..."
        continue
    fi
    
    accelerate launch \
        --gpu_ids 2 \
        only_eval.py \
        --config_file "config/test_dpr_${dataset}_eval.yaml"
done