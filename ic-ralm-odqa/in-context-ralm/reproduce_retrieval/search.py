# Perform retrieval by DPR and pre-built index
# Ref: https://github.com/castorini/pyserini/blob/master/docs/usage-search.md#learned-dense-retrieval-models
# from pyserini.search.faiss import FaissSearcher, DprQueryEncoder

# # encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
# encoder = DprQueryEncoder('facebook/dpr-question_encoder-single-nq-base')
# searcher = FaissSearcher.from_prebuilt_index(
#     # 'msmarco-passage-tct_colbert-hnsw',
#     'wikipedia-dpr-100w.dpr-single-nq',
#     encoder
# )
# hits = searcher.search('who got the first nobel prize in physics')

# for i in range(0, 10):
#     print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# # %%
# import json
# from pyserini.search.lucene import LuceneImpactSearcher

# # %%
# # TODO: process dataset
# with open("../data/nq-test-subset.json", "r") as f:
#     dataset = json.load(f)

# # %%
# nq_ctxs = []
# for example in dataset:
#     # example["ctxs"] is a list of dicts with key "text"
#     nq_ctxs.extend([ctx["text"] for ctx in example["ctxs"]])

# # %%
# questions = [example["question"] for example in dataset]
# answers = [example["answers"] for example in dataset]
# all_hits = []
# searcher = LuceneImpactSearcher.from_prebuilt_index(
#     'msmarco-v1-passage-unicoil', # learned sparse index
#     'castorini/unicoil-msmarco-passage')

# for question in questions:
#     hits = searcher.search(question)
#     all_hits.append([{"docid": hit.docid, "score": hit.score} for hit in hits])

# %%

# hits = searcher.search('what is a lobster roll?')

# for i in range(0, 10):
    # print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2-passage')

# %%
doc = searcher.doc('7157715')

# print(doc.contents())
print(doc.raw()) # only either of contents or raw will work, depending on the index

import json
json_doc = json.loads(doc.raw())

json_doc['contents']

# # fetch all documents in the all_hits
# # %%
# msmarco_ctxs = []
# for hits in all_hits:
#     for hit in hits:
#         doc = searcher.doc(hit["docid"])
#         json_doc = json.loads(doc.raw())
#         msmarco_ctxs.append(json_doc['contents'])

# # %%
# # search if any doc in msmarco_ctxs appears in nq_ctxs
# # Initialize an empty list to store the results
# docs_in_nq_ctxs = []

# # Iterate over msmarco_ctxs
# for doc in msmarco_ctxs:
#     # Check if doc is in nq_ctxs
#     if doc in nq_ctxs:
#         # If it is, append it to the results list
#         docs_in_nq_ctxs.append(doc)

# # Print the results
# print(docs_in_nq_ctxs)

# # %%
# mean_len_nq_ctxs = sum([len(ctx) for ctx in nq_ctxs]) / len(nq_ctxs)
# mean_len_msmarco_docs = sum([len(doc) for doc in msmarco_ctxs]) / len(msmarco_ctxs)

# print(f"Mean length of NQ contexts: {mean_len_nq_ctxs}")
# print(f"Mean length of MSMARCO documents: {mean_len_msmarco_docs}")
# # %%
# from difflib import SequenceMatcher

# # Function to find common substrings
# def common_substrings(string1, string2):
#     matcher = SequenceMatcher(None, string1, string2)
#     match_blocks = matcher.get_matching_blocks()
#     common_subs = []
#     for block in match_blocks:
#         common_subs.append(string1[block[0]:block[0]+block[2]])
#     return common_subs

# # Initialize an empty list to store the results
# common_substrings_list = []

# # Iterate over the first 10 contexts in msmarco_ctxs
# for msmarco_doc in msmarco_ctxs[:10]:
#     # Iterate over all contexts in nq_ctxs
#     for nq_doc in nq_ctxs:
#         # Find common substrings between msmarco_doc and nq_doc
#         common_subs = common_substrings(msmarco_doc, nq_doc)
#         # If there are any common substrings, append them to the results list
#         if common_subs:
#             common_substrings_list.append(common_subs)

# # Print the results
# for i, common_subs in enumerate(common_substrings_list, 1):
#     print(f"Common substrings between MSMARCO doc {i} and NQ contexts:")
#     for sub in common_subs:
#         print(sub)
# # %%
# # write msmarco contexts to a file with questions
# # each question is paired with 10 ctxs in msmarco_ctxs
# # it should be a list of dicts with keys "question" and "ctxs"

# # Initialize an empty list to store the dictionaries
# data = []

# # Iterate over the questions and answers
# for i, (question, answer) in enumerate(zip(questions, answers)):
#     # Create a dictionary with the question and the first 10 contexts
#     # The first 10 contexts are the first 10 documents in msmarco_ctxs
#     d = {"question": question, "answers": answer, 
#          "msmarco_ctxs": msmarco_ctxs[i*10:(i+1)*10], "nq_ctxs": nq_ctxs[i*10:(i+1)*10]}
#     data.append(d)
# # %%
# # Now you can write the data to a file
# with open('../data/sanitycheck_ctxs_nq-test-subset.json', 'w') as f:
#     json.dump(data, f, indent=4)
        

# # %%

# %%
