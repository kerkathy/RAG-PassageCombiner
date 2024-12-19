# Perform FAISS retrieval from existing corpus (index) and convert to DPR retrieval run 
# for a custom dataset and corpus.
# Usage: bash generate_custom_retrieval.sh <corpus_name>

set -e
export CUDA_VISIBLE_DEVICES=1

dataset_short_name=msmarcoqa
dataset_path=/home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/data/msmarco-qa/MSMARCO-Question-Answering/Data/train_v2.1_nlgen-subset-12467.json
topic_reader=io.anserini.search.topicreader.DprNqTopicReader
encoder_name=facebook/dpr-question_encoder-multiset-base

corpus_name=$1
# corpus_name=wiki
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
output_file=result/${dataset_short_name}/${corpus_name}.${dataset_short_name}.hits-100.json
formatted_output_file=result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
intermediate_file=${output_file/.json/.txt}
topic_file_intid=${output_file/.json/.topic.int-id.tsv}
topic_file_qa=${output_file/.json/.topic.qa.tsv}
# mapping_file=${output_file/.json/.mapping.json}

echo "----------------------------------------"
printf "%-25s\t%s\n" "CUDA_VISIBLE_DEVICES:" "$CUDA_VISIBLE_DEVICES"
printf "%-25s\t%s\n" "dataset_short_name:" "$dataset_short_name"
printf "%-25s\t%s\n" "dataset_path:" "$dataset_path"
printf "%-25s\t%s\n" "topic_reader:" "$topic_reader"
printf "%-25s\t%s\n" "encoder_name:" "$encoder_name"
printf "%-25s\t%s\n" "corpus_name:" "$corpus_name"
printf "%-25s\t%s\n" "faiss_index_suffix:" "$faiss_index_suffix"
printf "%-25s\t%s\n" "faiss_index_dir:" "$faiss_index_dir"
printf "%-25s\t%s\n" "luc_index_dir:" "$luc_index_dir"
printf "%-25s\t%s\n" "output_file:" "$output_file"
printf "%-25s\t%s\n" "formatted_output_file:" "$formatted_output_file"
printf "%-25s\t%s\n" "intermediate_file:" "$intermediate_file"
printf "%-25s\t%s\n" "topic_file_intid:" "$topic_file_intid"
printf "%-25s\t%s\n" "topic_file_qa:" "$topic_file_qa"
echo "----------------------------------------"


if [ ! -d $faiss_index_dir ] || [ ! -d $luc_index_dir ]; then
    echo "Directory $faiss_index_dir or $luc_index_dir does not exist."
    exit 1
fi

if [ -f $formatted_output_file ]; then
# if [ -f $output_file ] || [ -f $formatted_output_file ]; then
# if [ -f $output_file ] || [ -f $formatted_output_file ] || [ -f $intermediate_file ]; then
  echo "File $output_file or $formatted_output_file or $intermediate_file already exists."
  exit 1
fi

# Convert raw dataset file to topics file
python convert_raw_to_topic.py \
    --input $dataset_path \
    --output $topic_file_intid \
    --dataset $dataset_short_name \
    --function topic_conversion \
    --format int-id

echo "Finished converting raw dataset to topics file"

# topic.tsv format: <qid> <query>
## Method 1. Use Pyserini to search and convert to DPR retrieval run
python -m pyserini.search.faiss \
  --threads 16 --batch-size 512 \
  --index $faiss_index_dir \
  --encoder $encoder_name \
  --topics $topic_file_intid \
  --output $intermediate_file \
  --hits 100

# echo "Finished faiss search for $topic_file"

# Create a tsv with <question>\t<answer>
python convert_raw_to_topic.py \
    --input $dataset_path \
    --output $topic_file_qa \
    --dataset $dataset_short_name \
    --function topic_conversion \
    --format qa

# For corpus whose contents has title, e.g., Wiki (DPR) corpus, use this standard run
# python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
# For corpus whose contents has no title, e.g., (wiki+web) or (web) corpus, use this custom run
python convert_trec_run_to_dpr_retrieval_run.py \
  --topics-file $topic_file_qa \
  --topics-reader $topic_reader \
  --index $luc_index_dir \
  --input $intermediate_file \
  --output $output_file \
  --store-raw

echo "Finished converting trec run to dpr retrieval run"
echo "File written to $output_file"


# only for msmarcoQA, we need query_id for evaluation
if [ $dataset_short_name == "msmarcoqa" ]; then
  python format.py \
      --input_file $output_file \
      --output_file $formatted_output_file \
      --have_raw \
      --add_id \
      --id_dataset $dataset_path
else
  python format.py \
      --input_file $output_file \
      --output_file $formatted_output_file \
      --have_raw
fi

echo "Finished format.py, added raw text for $formatted_output_file"

echo "***Finish generate_custom_retrieval.sh for [$dataset_short_name] dataset on [$corpus_name] corpus***"