export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=0

# Remember to change the dataset_path and output_dir
# dataset_path="reproduce_retrieval/result/formatted-web.hotpotqa-dev.hits-100.json"
dataset_short_name=msmarcoqa
# dataset_short_name=nq-test
# dataset_short_name=$1
# "nq-test" "dpr-trivia-test" "hotpot" 
# main_name="trivia-test"
# main_name="my_orig_para1-nq-test"

num_docs=2

# corpus_name=wiki
# corpus_name=web
# corpus_name=wiki-web
corpus_name=ms2
dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
output_dir="output_${corpus_name}_${dataset_short_name}_docs_${num_docs}"

# for threshold in 0.5 0.6 0.7 0.8 0.9; do
#   for method in "tfidf"; do
#   for method in "tfidf" "spacy"; do
    # dataset_path=data/msmarco-unicoil.nq-test.hits-100.json
    # output_dir=output_msmarco-unicoil-nq-test

    # reranked_file=${dataset_path/.json/-reranked-${method}-${threshold}.json}
    # reranked_output_dir=${output_dir}/reranked_${method}-${threshold}

    # every 10 min, check if dataset_path exists
    # while [ ! -f $dataset_path ]; do
    #   echo "File $dataset_path not found. Sleeping for 10 min."
    #   sleep 10m
    # done
    
    python eval_qa.py \
    --model_name huggyllama/llama-7b \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --num_docs $num_docs \
    --cache_dir cache

    # python eval_qa.py \
    # --model_name huggyllama/llama-7b \
    # --dataset_path $reranked_file \
    # --output_dir $reranked_output_dir \
    # --num_docs $num_docs \
    # --cache_dir cache
#   done
# done