# Usage: python tsv_to_jsonl.py --split 26 [--subset] [--paraphrase]
# n = 1 to 40

for i in {4..40}
do
    python tsv_to_jsonl.py --split $i --paraphrase 3
    if [ $? -ne 0 ]; then
        echo "Command failed with exit status $?. Exiting loop."
        break
    fi
    echo "Finish converting split $i to jsonl."
done
