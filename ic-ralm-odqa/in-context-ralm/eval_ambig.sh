# Converts the output of the model to the format required by the AmbigQA evaluation script 
# and runs the evaluation script.
# Usage: ./eval_ambig.sh

set -e

ambig_dir=data/AmbigQA

for dir in $(find output_ms2_AmbigQA_docs_2/ -maxdepth 1 -mindepth 1 -type d)
do
    echo "Running script for $dir"
    old_path=${dir}/prediction.json
    ref_path=${ambig_dir}/data/dev_light_indent.json
    new_path=${dir}/processed_prediction.json
    result_path=${dir}/eval.txt

    if [ ! -f ${old_path} ] || [ -f ${result_path} ]; then
        echo "Skipping $dir"
        continue
    fi

    python format_ambig_ans.py \
        --old ${old_path} \
        --ref ${ref_path} \
        --new ${new_path}

    python ${ambig_dir}/ambigqa_evaluate_script.py \
        --reference_path ${ref_path} \
        --prediction_path ${new_path} &> ${result_path}
    
    echo "Result saved to ${result_path}"
    echo ""
done