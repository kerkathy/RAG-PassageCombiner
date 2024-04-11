corpus_name="ms2"
dataset_name="msmarcoqa"
echo "corpus: $corpus_name"
echo "dataset: $dataset_name"

for num_doc in {1..10} ; do
    dir_name=/home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/output_${corpus_name}_${dataset_name}_docs_${num_doc}/reranked_tfidf-0.5/
    # dir_name=/home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/output_ms2_msmarcoqa_docs_5/reranked_tfidf-${threshold}/
    echo "num_doc: $num_doc"
    echo "dir_name: $dir_name\n"

    ./run.sh \
        ${dir_name}gold_answers.json \
        ${dir_name}prediction.json
    echo "----------------------------------------"
done