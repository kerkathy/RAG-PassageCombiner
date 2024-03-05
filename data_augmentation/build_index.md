# Notes for Building Index
## Data Preparation
### NQ
Currently the data should be in the `DPR/dpr/resource/downloadsDPR/dpr/resource/downloads` folder. There're many splits:
* original: the 2018 dpr corpus used for DPR training, 21015324 is the max id. While the text of the corpus can be downloaded from huggingface or dpr repo, the prebuilt index can directly be downloaded from pyserini. 
* paraphrase: each is a paragraph paraphrased from one of the original corpus, and the id is set to be starting with 21015325.

In the future, maybe there'll be paraphrase_2, paraphrase_3, ...

## Builing Index
### Lucene
Haven't found a way of combining multiple small indices, so a temporary workaround is to build a lucene index for all data (original + paraphrase) altogether. To build a index, put the folder of specific jsonl file in `--input` argument, and set the destination directory at `--index`. 
```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ../DPR/dpr/resource/downloads/data/all \
  --index lucene_index/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
```

p.s. building Lucene index takes really short time! It took me 16 mins to build an index with 40M documents with 8 threads. 

### Faiss
A good thing is that Faiss index can be combined, so let's build the indices first. 
* The `--corpus` can be either be a json file, or a directory that contains multiple json files
* The `--fields` is the column contained (implicitly) in the contents field, seperated by `--delimiter`
* With `--to-faiss`, the generated embeddings will be stored as FaissIndexIP directly. Otherwise will have the vector saved.
* Use `--embeddings` to set output directory. Note that this code doesn't preserve the existing output directory, so make sure that is empty before launching the run. 
* `--fields` is the column we want to encode.
* Building faiss index needs GPU, set it at `--device`.
```
python -m pyserini.encode \
input     --corpus ../DPR/dpr/resource/downloads/data/wiki_split_para_jsonl/psgs_w100-${split}-paraphrase.jsonl \
          --fields title text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
output    --embeddings ./faiss_index/${split} \
          --to-faiss \
encoder   --encoder facebook/dpr-ctx_encoder-single-nq-base \
          --fields text \
          --batch 32 \
          --device cuda:${cuda} \
          --fp16
```

After building some indices, we can combine all indices in a specific directory.
* `--prefix` means the prefix of all indices we want to combine. Make sure a `<prefix>/0` directory exists with index!
* `--shard-num` is the number of indices we want to combine.
The code below combines indices in directory `<prefix>/0`, `<prefix>/1` to `<prefix>/<shard_num-1>`
```
python -m pyserini.index.merge_faiss_indexes \
    --prefix faiss_index/ \
    --shard-num 2
```

### My Notes :P
* `faiss_index/all/0`: index of original corpus. Largest id is 21015324.
* `faiss_index/all/1`: copied from the index that contains all paraphrases generated in the 1st round, i.e. the one at `faiss_index/para/full`. Documents here have id starting with 21015325.
* `lucene_index_orig_para1`: index of original (wiki 2018 snapshot; dpr) + paraphrases from 1st round
* `lucene_index_wiki_msmc`: index of original (wiki 2018 snapshot; dpr; size = 21015324) + msmarco passage corpus (size = 8,841,823). Ids of msmarco passages are given from 21,015,325 to (21,015,324 + 8,841,823).