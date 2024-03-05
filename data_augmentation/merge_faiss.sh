# Create a merged index for the paragraph index at directory faiss_index/para/full
prefix_dir="faiss_index/wiki_web/"
target_dir=$prefix_dir"full"

# check the target directory to see if it already exists
if [ -d $target_dir ]; then
    echo "Directory $target_dir already exists."
    exit 1
fi

echo "Merging indexes in $prefix_dir to $target_dir"

python -m pyserini.index.merge_faiss_indexes \
    --prefix $prefix_dir \
    --shard-num 2