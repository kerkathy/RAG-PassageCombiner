# run diversity_rerank.py for diff threshold and method
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=1

corpus_name=ms2
# input_file=data/msmarco-unicoil-nq-test.json

for dataset_short_name in "dpr-trivia-test" "hotpot"; do
  for method in "tfidf" "spacy"; do
    dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
    python diversity_rerank.py \
      --input_file $dataset_path \
      --sim_thresholds 0.5 0.6 0.7 0.8 0.9 \
      --sim_method $method
  done
done
