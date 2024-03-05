# Generate retrieval from MSMARCO corpus

set -e

print_variables() {
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
    printf "%-25s\t%s\n" "topic_file:" "$topic_file"
    echo "----------------------------------------"
}

for dataset_short_name in "msmarcoqa"; do
# for dataset_short_name in "dpr-trivia-test" "hotpot"; do
    echo "Running retrieval for $dataset_short_name"
    corpus_name=ms2
    output_file=result/${dataset_short_name}/${corpus_name}.${dataset_short_name}.hits-100.json
    intermediate_file=${output_file/.json/.txt}
    topic_reader=io.anserini.search.topicreader.DprNqTopicReader
    formatted_output_file=result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json

    # if [ -f $intermediate_file ]; then
    #     echo "Output file already exists, skipping"
    #     continue
    # fi
    # if [ -f $output_file ]; then
    #     echo "Output file $output_file already exists, skipping"
    #     continue
    # fi

    if [ $dataset_short_name == "hotpot" ]|| [ $dataset_short_name == "msmarcoqa" ]; then
        dataset_path=/home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/data/msmarco-qa/MSMARCO-Question-Answering/Data/train_v2.1_nlgen-subset-12467.json
        topic_file_intid=${output_file/.json/.topic.int-id.tsv}
        topic_file_qa=${output_file/.json/.topic.qa.tsv}
    
        print_variables

        # Convert raw dataset file to topics file
        python convert_raw_to_topic.py \
            --input $dataset_path \
            --output $topic_file_intid \
            --dataset $dataset_short_name \
            --function topic_conversion \
            --format int-id

        echo "Finished converting raw dataset to topics file"

        # topic.tsv format: <qid> <query>
        python -m pyserini.search.lucene \
            --index msmarco-v2-passage \
            --topics $topic_file_intid \
            --output $intermediate_file \
            --batch-size 36 --threads 12 \
            --hits 100 \
            --bm25

        # Create a tsv with <question>\t<answer>
        python convert_raw_to_topic.py \
            --input $dataset_path \
            --output $topic_file_qa \
            --dataset $dataset_short_name \
            --function topic_conversion \
            --format qa
        
        # For corpus whose contents has no title, e.g., (wiki+web) or (web) corpus, use this custom run
        python convert_trec_run_to_dpr_retrieval_run.py \
            --topics-file $topic_file_qa \
            --topics-reader $topic_reader \
            --index msmarco-v2-passage \
            --input $intermediate_file \
            --output $output_file \
            --combine-title-text \
            --store-raw

        echo "Finished converting trec run to dpr retrieval run"
        echo "File written to $output_file"

    elif [ $dataset_short_name == "nq-test" ] || [ $dataset_short_name == "dpr-trivia-test" ]; then
        topic_file=$dataset_short_name
        
        print_variables
        
        # Create a tsv with <question>\t<answer>
        python -m pyserini.search.lucene \
        --index msmarco-v2-passage \
        --topics $dataset_short_name \
        --output $intermediate_file \
        --batch-size 36 --threads 12 \
        --hits 100 \
        --bm25

        # For corpus whose contents has no title, e.g., (wiki+web) or (web) corpus, use this custom run
        python convert_trec_run_to_dpr_retrieval_run.py \
            --topics $topic_file \
            --index msmarco-v2-passage \
            --input $intermediate_file \
            --output $output_file \
            --store-raw

        echo "Finished converting trec run to dpr retrieval run"
        echo "File written to $output_file"
    fi
    
    echo "Finished retrieval for $dataset_short_name, output file: $output_file"

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

done