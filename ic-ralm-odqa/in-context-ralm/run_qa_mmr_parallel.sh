set -e

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="4,5"
# export CUDA_VISIBLE_DEVICES="1,2,4,5"
# export CUDA_VISIBLE_DEVICES="0,1,4,5"
# export CUDA_VISIBLE_DEVICES="2,3,6,7,8"
# export CUDA_VISIBLE_DEVICES="0,3"

# Remember to change the dataset_path and output_dir
# dataset_path="reproduce_retrieval/result/formatted-web.hotpotqa-dev.hits-100.json"
dataset_short_name=$1
# "nq-test" "dpr-trivia-test" "hotpot" "msmarcoqa"
# main_name="trivia-test" "my_orig_para1-nq-test"

# check cmd line argument
if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "Usage: $0 dataset_short_name"
  exit 1
fi

num_docs=2
max_token=10
if [[ "$dataset_short_name" == *"msmarcoqa"* ]]; then
  num_docs=5
  max_token=100
fi

# corpus_name=wiki
# corpus_name=web
# corpus_name=wiki-web
corpus_name=ms2
dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
output_dir="output_${corpus_name}_${dataset_short_name}_docs_${num_docs}"
lambda_param=0.5
threshold=1
method="tfidf"

# for lambda_param in 0.8 0.9; do
for lambda_param in 0.6 0.7 0.8 0.9; do
  reranked_file=${dataset_path/.json/-reranked-mmr-${lambda_param}-${method}-${threshold}.json}
  reranked_output_dir=${output_dir}/reranked-mmr-${lambda_param}-${method}-${threshold}

  # print all variable
  echo "----------------------------------------"
  printf "%-25s\t%s\n" "dataset_short_name:" "$dataset_short_name"
  printf "%-25s\t%s\n" "corpus_name:" "$corpus_name"
  printf "%-25s\t%s\n" "dataset_path:" "$reranked_file"
  printf "%-25s\t%s\n" "output_dir:" "$reranked_output_dir"
  printf "%-25s\t%s\n" "num_docs:" "$num_docs"
  printf "%-25s\t%s\n" "max_token:" "$max_token"
  echo "----------------------------------------"
  
  python eval_qa.py \
  --model_name huggyllama/llama-7b \
  --dataset_path $reranked_file \
  --output_dir $reranked_output_dir \
  --num_docs $num_docs \
  --cache_dir cache \
  --max_token $max_token \
  --model_parallelism
done