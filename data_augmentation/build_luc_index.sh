python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/msmarco-psg/collection/jsonl \
  --index lucene_index_wiki_web/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw