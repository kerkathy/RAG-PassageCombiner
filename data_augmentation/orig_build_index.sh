python -m pyserini.encode \
  input   --corpus tests/resources/simple_cacm_corpus.json \
          --fields text \
          --delimiter "\n" \
  output  --embeddings path/to/output/dir \
          --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
          --fields text
          --batch 32 \
          --fp16