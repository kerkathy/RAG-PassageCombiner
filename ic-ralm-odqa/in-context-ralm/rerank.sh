# run diversity_rerank.py for diff algo, filter method, and threshold 
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=0

corpus_name=ms2
# input_file=data/msmarco-unicoil-nq-test.json
sim_method=tfidf
algo=mmr

# for dataset_short_name in "msmarcoqa"; do
for dataset_short_name in "nq-test" "dpr-trivia-test" "hotpot" "msmarcoqa"; do
  for lambda_param in 0.6 0.7 0.8 0.9; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    python diversity_rerank.py \
      --input_file $dataset_path \
      --sim_method $sim_method \
      --algo $algo \
      --lambda_param $lambda_param
  done
done
