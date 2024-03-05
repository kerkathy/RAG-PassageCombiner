"""
View statistics of the index we built.
Check two faiss indexes: one for the original corpus and one for the paraphrased corpus.
And one Lucene index for all corpus (original + paraphrased).

1. Number of documents: 
faiss_original = faiss_paraphrased
faiss_original + faiss_paraphrased = lucene_all
"""
# %%
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher

# %%
lucene_index_dir = "lucene_index_wiki_web"
luc_searcher = LuceneSearcher(lucene_index_dir)
print("Total Lucene:", luc_searcher.num_docs)

# %%
faiss_index_orig = "faiss_index/wiki_web/full"
# faiss_index_orig = "faiss_index/all/0"
# faiss_index_para = "faiss_index/all/1"

faiss_searcher_orig = FaissSearcher(
    faiss_index_orig,
    'facebook/dpr-question_encoder-single-nq-base'
)
# %%
print("Faiss index original:", faiss_searcher_orig.num_docs)
# print("Faiss index paraphrased:", faiss_searcher_para.num_docs)
# print("Total Faiss index:",faiss_searcher_orig.num_docs + faiss_searcher_para.num_docs)

# %%
# Check each faiss index
sub_faiss_dir = "faiss_index/para/"
splits = 40
size_list = []
print("Faiss splits:")
for i in range(splits):
    faiss_searcher = FaissSearcher(
        f"{sub_faiss_dir}{i}",
        'facebook/dpr-question_encoder-single-nq-base'
    )
    print(f"Split {i}: {faiss_searcher.num_docs}")
    size_list.append(faiss_searcher.num_docs)

print("Total Faiss splits:", sum(size_list))
# %%
query = "What is the capital of France?"
hits = faiss_searcher_orig.search(query, 10)
# hits = faiss_searcher_para.search(query, 10)
# %%
# Fetch raw text
import json
ctxs = []

for hit in hits:
    print(hit.docid, hit.score)
    # print(hit["docid"], hit["score"])
    doc = luc_searcher.doc(hit.docid)
    # doc = luc_searcher.doc(hit["docid"])
    json_doc = json.loads(doc.raw())
    print(json_doc['contents'])
    ctxs.append(json_doc['contents'])
# %%
