# when i = 1, then round = 2
# when i = 2 or 3, then round = 1

# accelerate launch \
#     --gpu_ids 1 \
#     only_eval_with_docid.py \
#     --config_file "config/test_dpr_${dataset}_${i}_${round}r.yaml"

for dataset in hotpot trivia nq
do
    for i in {1..3}
    do
        if [ $i -eq 1 ]; then
            round=2
        else
            round=1
        fi
        config_file="config/test_dpr_${dataset}_${i}_${round}r_eval.yaml"
        echo "Running $dataset $i $round"
            if [ ! -f $config_file ]; then
                echo "$config_file does not exist. Skipping..."
                continue
            fi
        echo "Running $config_file"
        accelerate launch \
            --gpu_ids 2 \
            only_eval_with_docid.py \
            --config_file "config/test_dpr_${dataset}_${i}_${round}r_eval.yaml"
    done
done