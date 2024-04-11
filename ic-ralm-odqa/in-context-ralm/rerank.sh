set -e
# run diversity_rerank.py for diff algo, filter method, and threshold 
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="2"

corpus_name=ms2

dataset_short_name=$1
algo=$2
sim_method=$3
# "nq-test" "dpr-trivia-test" "hotpot" "msmarcoqa" "eli5" "strategyQA" "AmbigQA"
# algo=mmr basic
# sim_method=sbert tfidf

start_time=$(date +%s)

if [ "$algo" = "mmr" ]; then
  # mmr
  for dataset_short_name in $dataset_short_name; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    # python diversity_rerank.py \
    python tmp_diversity_rerank.py \
      --input_file $dataset_path \
      --sim_method $sim_method \
      --algo $algo \
      --lambda_params 0.5 0.6 0.7 0.8 0.9 1.0
  done
elif [ "$algo" = "basic" ]; then
  # basic
  for dataset_short_name in $dataset_short_name; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    # python diversity_rerank.py \
    python tmp_diversity_rerank.py \
      --input_file $dataset_path \
      --sim_method $sim_method \
      --algo $algo \
      --sim_thresholds 0.5 0.6 0.7 0.8 0.9
  done
else 
  echo "Invalid algo"
  exit 1
fi

end_time=$(date +%s)
execution_time=$(expr $end_time - $start_time)

echo "Finish"
echo "Execution time: $execution_time seconds"