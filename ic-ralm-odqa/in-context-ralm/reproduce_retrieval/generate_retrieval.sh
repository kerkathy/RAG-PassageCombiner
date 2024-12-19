# Perform FAISS retrieval on datasets **provided by Pyserini** e.g., nq, trivia-test
# export CUDA_VISIBLE_DEVICES=5
# Usage: bash generate_retrieval.sh <corpus_name>

set -e

dataset_file=hotpot/intid_hotpot_dev_distractor_subset_topic.tsv
dataset_name=hotpot-dev
# dataset_name=nq-test
# dataset_name=hotpotqa-dev

corpus_name=wiki-web
# corpus_name=web
# corpus_name=wiki-web

if [ $corpus_name == "wiki" ]; then
  faiss_index_suffix="0"
elif [ $corpus_name == "web" ]; then
  faiss_index_suffix="1"
elif [ $corpus_name == "wiki-web" ]; then
  faiss_index_suffix="full"
else
  echo "Corpus name $corpus_name not recognized."
  exit 1
fi

# faiss_index_dir=/home/guest/r11944026/research/data_augmentation/faiss_index/wiki_web/full
faiss_index_dir=/home/guest/r11944026/research/data_augmentation/faiss_index/wiki_web/$faiss_index_suffix
luc_index_dir=/home/guest/r11944026/research/data_augmentation/lucene_index_wiki_web
output_file=result/${corpus_name}.${dataset_name}.hits-100.json
# output_file=result/web.trivia-test.hits-100.json
formatted_output_file=result/formatted-${corpus_name}.${dataset_name}.hits-100.json
encoder_name=facebook/dpr-question_encoder-multiset-base

# intermediate_file is output_file replaced with .txt
intermediate_file=${output_file/.json/.txt}

if [ ! -d $faiss_index_dir ] || [ ! -d $luc_index_dir ]; then
    echo "Directory $faiss_index_dir or $luc_index_dir does not exist."
    exit 1
fi &&

if [ -f $output_file ] || [ -f $formatted_output_file ]; then
# if [ -f $output_file ] || [ -f $formatted_output_file ] || [ -f $intermediate_file ]; then
  echo "File $output_file or $formatted_output_file or $intermediate_file already exists."
  exit 1
fi

### Method 1. Use Pyserini to search and convert to DPR retrieval run

# python -m pyserini.search.faiss \
#   --threads 16 --batch-size 512 \
#   --index $faiss_index_dir \
#   --encoder $encoder_name \
#   --topics $dataset_name \
#   --output $intermediate_file \
#   --hits 100

# echo "Finished faiss search for $dataset_name"

# For (wiki+web) or (web) corpus, use this custom run
# python convert_trec_run_to_dpr_retrieval_run.py \
# For pure Wiki (DPR) corpus, use this standard run
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
  --topics $dataset_name \
  --index $luc_index_dir \
  --input $intermediate_file \
  --output $output_file \
  --store-raw

# echo "Finished converting trec run to dpr retrieval run"
# echo "File written to $output_file"

# python format.py \
#     --input_file $output_file \
#     --output_file $formatted_output_file \
#     # --have_raw

# echo "Finished search.py, added raw text for $formatted_output_file"

echo "Finish"