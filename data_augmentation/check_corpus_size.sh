# for i = 1 to 40
# count how many lines in the file psgs_w100-$i.tsv
# and calculate the sum of all lines in all files
# output all results to a file called psgs_w100_sizes.txt
# count the number of lines in each file
file_name="sizes_jsonl_para.txt"

if [ -f $file_name ]; then
    echo "File $file_name already exists."
    exit 1
fi

for i in {1..40}
do
    wc -l ~/research/DPR/dpr/resource/downloads/data/all_jsonl/psgs_w100-$i-paraphrase.jsonl
done >> $file_name


# calculate the sum of all lines in all files
echo "\ntotal:" >> $file_name
awk '{s+=$1} END {print s}' $file_name >> $file_name

echo "File $file_name created."