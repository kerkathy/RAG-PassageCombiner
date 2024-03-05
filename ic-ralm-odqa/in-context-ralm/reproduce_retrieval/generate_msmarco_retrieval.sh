export CUDA_VISIBLE_DEVICES=2

# faiss_index_dir=/home/guest/r11944026/research/data_augmentation/faiss_index/all/full
# luc_index_dir=msmarco-v1-passage
# luc_index_dir=msmarco-v1-passage-unicoil
# luc_index_dir=/home/guest/r11944026/research/data_augmentation/lucene_index_orig_para1
output_file=msmarco-unicoil.nq-test.hits-100.txt
output_json=${output_file%.txt}.json
# output_file=result/myindex.nq-test.hits-100.json
dataset_name=nq-test

# python -m pyserini.search.faiss \
#   --threads 16 --batch-size 512 \
#   --index msmarco-v1-passage.tct_colbert-v2-hnp \
#   --topics msmarco-passage-dev-subset \
#   --encoder castorini/tct_colbert-v2-hnp-msmarco \
#   --output run.msmarco-v1-passage.tct_colbert-v2-hnp-avg-prf-pytorch.dev.txt \
#   --prf-method avg --prf-depth 3

# python -m pyserini.search.lucene \
#   --threads 16 --batch-size 128 \
#   --index msmarco-v1-passage-unicoil \
#   --impact \
#   --topics $dataset_name \
#   --output $output_file \
#   --hits 100

echo "Converting MSMARCO run $output_file to $output_json"
python convert_trec_run_to_dpr_retrieval.py \
  --topics $dataset_name \
  --index msmarco-v1-passage \
  --input $output_file \
  --output $output_json \
  --store-raw


# echo "Finished search and saved to $output_file"