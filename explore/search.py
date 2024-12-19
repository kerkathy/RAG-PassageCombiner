import json
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
import os
import random
from tqdm import tqdm

# Set environment variables for GPU (if applicable)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Function to load a dataset from a JSON file
def load_dataset(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Function to save data to a JSON file
def save_dataset(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {output_path}")

# Function to perform retrieval using DPR with a prebuilt index
def dpr_retrieval(query, index_name='wikipedia-dpr-100w.dpr-single-nq'):
    encoder = DprQueryEncoder('facebook/dpr-question_encoder-single-nq-base')
    searcher = FaissSearcher.from_prebuilt_index(index_name, encoder)
    hits = searcher.search(query)
    return [{"docid": hit.docid, "score": hit.score} for hit in hits]

# Function to perform retrieval using Lucene Impact Searcher
def lucene_retrieval(queries, index_name='msmarco-v1-passage-unicoil', encoder_name='castorini/unicoil-msmarco-passage'):
    searcher = LuceneImpactSearcher.from_prebuilt_index(index_name, encoder_name)
    results = []
    for query in queries:
        hits = searcher.search(query)
        results.append([{"docid": hit.docid, "score": hit.score} for hit in hits])
    return results

# Function to extract document content from search results
def fetch_documents(searcher, hits):
    documents = []
    for hit in hits:
        doc = searcher.doc(hit["docid"])
        if doc:
            json_doc = json.loads(doc.raw())
            documents.append(json_doc.get('contents', ''))
    return documents

# Function to analyze common substrings between two lists of texts
def common_substrings(string1, string2):
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, string1, string2)
    match_blocks = matcher.get_matching_blocks()
    return [string1[block[0]:block[0] + block[2]] for block in match_blocks]

# Function to add a key to nested dictionaries
def add_key_to_contexts(data, key):
    for example in data:
        for ctx in example.get("ctxs", []):
            ctx[key] = None
    return data

# Function to sample and reformat a dataset
def sample_and_reformat_dataset(data, k, seed=42):
    random.seed(seed)
    all_ids = list(data["query"].keys())
    random.shuffle(all_ids)
    sampled_ids = all_ids[:k]
3
    output_list = []
    for _id in sampled_ids:
        output_list.append({
            "_id": _id,
            "question": data["query"][_id],
            "query_type": data["query_type"][_id],
            "answer": data["answers"][_id]
        })

    return output_list

# Example Usage
if __name__ == "__main__":
    # Load example dataset
    dataset_path = "../data/nq-test-subset.json"
    dataset = load_dataset(dataset_path)

    # Perform retrieval
    queries = [example["question"] for example in dataset]
    answers = [example["answers"] for example in dataset]

    # Retrieve with Lucene
    lucene_results = lucene_retrieval(queries)

    # Extract documents
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2-passage')
    for query, result in zip(queries, lucene_results):
        docs = fetch_documents(searcher, result)
        print(f"Query: {query}\nDocs: {docs[:2]}\n")

    # Add a key "title" with None values to contexts
    updated_dataset = add_key_to_contexts(dataset, "title")

    # Save updated dataset
    save_dataset(updated_dataset, "../data/updated_nq-test-subset.json")

    # Sample and reformat large dataset
    input_path = "/home/guest/research/msmarco-data.json"
    data = load_dataset(input_path)
    sampled_data = sample_and_reformat_dataset(data, k=1000)
    save_dataset(sampled_data, "../data/msmarco_sample.json")
