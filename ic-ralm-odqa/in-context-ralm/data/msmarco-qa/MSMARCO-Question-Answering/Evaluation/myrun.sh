for corpus_name in "wiki" "web" "wiki-web" "ms2"; do
    echo "corpus: $corpus_name"
    ./run.sh \
        /home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/output_${corpus_name}_msmarcoqa_docs_5/gold_answers.json \
        /home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/output_${corpus_name}_msmarcoqa_docs_5/prediction.json
    echo "----------------------------------------"
done