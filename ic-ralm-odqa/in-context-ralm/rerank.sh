set -e
# run diversity_rerank.py for diff algo, filter method, and threshold 
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="2"

corpus_name=ms2

dataset_short_name=$1
algo=$2
sim_method=$3
# "nq-test" "dpr-trivia-test" "hotpot" "msmarcoqa" "eli5" "strategyQA" "AmbigQA"
# algo=basic mmr kmeans
# sim_method=sbert tfidf

start_time=$(date +%s)

if [ "$algo" = "mmr" ]; then
  for dataset_short_name in $dataset_short_name; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    python diversity_rerank.py \
      --input_file $dataset_path \
      --sim_method $sim_method \
      --algo $algo \
      --lambda_params 0.5 0.6 0.7 0.8 0.9 1.0
  done
elif [ "$algo" = "basic" ]; then
  for dataset_short_name in $dataset_short_name; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    python diversity_rerank.py \
      --input_file $dataset_path \
      --sim_method $sim_method \
      --algo $algo \
      --sim_thresholds 0.5 0.6 0.7 0.8 0.9
  done
elif [ "$algo" = "kmeans" ]; then
  for dataset_short_name in $dataset_short_name; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    python diversity_rerank.py \
      --input_file $dataset_path \
      --sim_method $sim_method \
      --algo $algo \
      --k 2 4 \
      # --debug
  done
else 
  echo "Invalid algo"
  exit 1
fi

end_time=$(date +%s)
execution_time=$(expr $end_time - $start_time)

echo "Finish"
echo "Execution time: $execution_time seconds"