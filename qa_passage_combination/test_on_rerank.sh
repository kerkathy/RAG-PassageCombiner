for dataset in nq trivia hotpot; do
    config_file="config/test_dpr_${dataset}_rerank_eval.yaml"
    echo "Running $config_file"

    if [ ! -f $config_file ]; then
        echo "$config_file does not exist. Skipping..."
        continue
    fi

    accelerate launch \
        --gpu_ids 2 \
        evaluate_on_reranked_validation.py \
        --config_file $config_file
done