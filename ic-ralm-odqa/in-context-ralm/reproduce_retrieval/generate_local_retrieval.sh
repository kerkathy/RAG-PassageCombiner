# Perform FAISS retrieval from existing local corpus (index) and convert to DPR retrieval run
# for query dataset from either Pyserini or a local dataset.
# Usage: bash generate_retrieval.sh <corpus_name> <dataset_type> <dataset_path_or_file> <dataset_name>

set -e

# Input parameters
corpus_name=$1 # Options: "wiki", "web", or "wiki-web"
dataset_type=$2  # Options: "local" or "pyserini"
dataset_path_or_file=$3 # path to local dataset or Pyserini dataset
dataset_name=$4 # E.g., "msmarcoqa", "nq", "trivia"

export CUDA_VISIBLE_DEVICES=1
encoder_name=facebook/dpr-question_encoder-multiset-base

# Determine FAISS index suffix based on corpus_name
case $corpus_name in
  "wiki")
    faiss_index_suffix="0"
    ;;
  "web")
    faiss_index_suffix="1"
    ;;
  "wiki-web")
    faiss_index_suffix="full"
    ;;
  *)
    echo "Error: Corpus name $corpus_name not recognized. Use 'wiki', 'web', or 'wiki-web'."
    exit 1
    ;;
esac

faiss_index_dir=/home/guest/r11944026/research/data_augmentation/faiss_index/wiki_web/$faiss_index_suffix
luc_index_dir=/home/guest/r11944026/research/data_augmentation/lucene_index_wiki_web
output_file=result/${corpus_name}.${dataset_name}.hits-100.json
formatted_output_file=result/formatted-${corpus_name}.${dataset_name}.hits-100.json
intermediate_file=${output_file/.json/.txt}

# Ensure directories exist
if [ ! -d $faiss_index_dir ] || [ ! -d $luc_index_dir ]; then
    echo "Error: Directory $faiss_index_dir or $luc_index_dir does not exist."
    exit 1
fi

# Check if output files already exist
if [ -f $output_file ] || [ -f $formatted_output_file ]; then
  echo "Error: File $output_file or $formatted_output_file already exists."
  exit 1
fi

# Display configuration
cat << EOF
----------------------------------------
CUDA_VISIBLE_DEVICES:   $CUDA_VISIBLE_DEVICES
corpus_name:           $corpus_name
dataset_type:          $dataset_type
dataset_path_or_file:  $dataset_path_or_file
dataset_name:          $dataset_name
faiss_index_dir:       $faiss_index_dir
luc_index_dir:         $luc_index_dir
output_file:           $output_file
formatted_output_file: $formatted_output_file
----------------------------------------
EOF

# Retrieval logic
if [ $dataset_type == "local" ]; then
    # Convert raw dataset to topics file
    topic_file_intid=${output_file/.json/.topic.int-id.tsv}
    python convert_raw_to_topic.py \
        --input $dataset_path_or_file \
        --output $topic_file_intid \
        --dataset $dataset_name \
        --function topic_conversion \
        --format int-id

    echo "Finished converting raw dataset to topics file"

    # topic.tsv format: <qid> <query>
    # FAISS retrieval
    python -m pyserini.search.faiss \
      --threads 16 --batch-size 512 \
      --index $faiss_index_dir \
      --encoder $encoder_name \
      --topics $topic_file_intid \
      --output $intermediate_file \
      --hits 100

    # Convert to DPR retrieval run
    topic_file_qa=${output_file/.json/.topic.qa.tsv}
    python convert_raw_to_topic.py \
        --input $dataset_path_or_file \
        --output $topic_file_qa \
        --dataset $dataset_name \
        --function topic_conversion \
        --format qa

    topic_reader=io.anserini.search.topicreader.DprNqTopicReader

    if [ $corpus_name == "wiki" ]; then
        python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
          --topics-file $topic_file_qa \
          --topics-reader $topic_reader \
          --index $luc_index_dir \
          --input $intermediate_file \
          --output $output_file \
          --store-raw
    else
        python convert_trec_run_to_dpr_retrieval_run.py \
          --topics-file $topic_file_qa \
          --topics-reader $topic_reader \
          --index $luc_index_dir \
          --input $intermediate_file \
          --output $output_file \
          --store-raw
    fi

    echo "Finished local dataset retrieval run"

elif [ $dataset_type == "pyserini" ]; then
    # FAISS retrieval
    python -m pyserini.search.faiss \
      --threads 16 --batch-size 512 \
      --index $faiss_index_dir \
      --encoder $encoder_name \
      --topics $topics_path \
      --output $intermediate_file \
      --hits 100

      # TODO add diff logic for different dataset

    if [ $corpus_name == "wiki" ]; then
        python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
            --topics $topics_path \
            --index $luc_index_dir \
            --input $intermediate_file \
            --output $output_file \
            --store-raw
    else
        python convert_trec_run_to_dpr_retrieval_run.py \
            --topics $topics_path \
            --index $luc_index_dir \
            --input $intermediate_file \
            --output $output_file \
            --store-raw
    fi

    echo "Finished Pyserini dataset retrieval run"

else
    echo "Error: Dataset type $dataset_type not recognized. Use 'local' or 'pyserini'."
    exit 1
fi

# Formatting
python format.py \
    --input_file $output_file \
    --output_file $formatted_output_file \
    --have_raw

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

echo "Finished formatting. Output at $formatted_output_file"
