# run the following command to build faiss index for the split
# bash build_faiss_index.sh 34 0
split=$1
cuda=$2

# check for correct num of args
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./build_faiss_index.sh <split> <cuda>"
    exit 1
fi

# check if faiss_index directory ./faiss_index/${split} exists
if [ -d "./faiss_index/$((${split}-1))" ]; then
    echo "Directory ./faiss_index/$((${split}-1)) exists."
    exit 1
fi

python -m pyserini.encode \
input     --corpus ../DPR/dpr/resource/downloads/data/all_jsonl/psgs_w100-${split}-paraphrase.jsonl \
          --fields title text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
output    --embeddings ./faiss_index/para/$((${split}-1)) \
          --to-faiss \
encoder   --encoder facebook/dpr-ctx_encoder-single-nq-base \
          --fields text \
          --batch 32 \
          --device cuda:${cuda} \
          --fp16 && \

echo "Indexing complete for split ${split} on cuda ${cuda}"