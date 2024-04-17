"""
Given a list of dictionaries, where each dict has keys "question", "answers", and "ctxs"
Rerank the ctxs based on diversity.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util

from ralm.file_utils import print_args

# global variables to store the frequently used variables
similarity_matrices = {}
query_similarity = {}
model = None

def eq_1(x):
    epsilon = 1e-9  # a very small number
    if abs(x - 1) < epsilon:
        return True
    else:
        return False


def custom_similarity(paragraphs, query=None, method="tfidf"):
    """
    Compute the similarity between paragraphs using the specified method.
    Args:
        paragraphs: list of strings
        query: list of a string
        method: str, one of "tfidf", "spacy"
    Returns:
        similarity_scores: 
            - list of lists (if query is None)
              where similarity_scores[i][j] is the similarity score between paragraph i and j
            - list of floats (if query is not None)
              where similarity_scores[i] is the similarity score between the query and paragraph i
    """
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
        score_matrix = vectorizer.fit_transform(paragraphs)
        if query is None:
            similarity_scores = cosine_similarity(score_matrix)
        else:
            query_vector = vectorizer.transform([query])
            similarity_scores = cosine_similarity(query_vector, score_matrix)
        return similarity_scores

    elif method == "spacy":
        if query is not None:
            raise NotImplementedError("Query is not supported for spacy similarity.")
        nlp = spacy.load("en_core_web_lg")
        docs = [nlp(text) for text in paragraphs]
        similarity_matrix = [[doc1.similarity(doc2) for doc2 in docs] for doc1 in docs]
    
    elif method == "sbert":
        global model
        passage_embedding = model.encode(paragraphs)
        
        if query is None:
            similarity_matrix = util.dot_score(passage_embedding, passage_embedding)
        else:
            query_embedding = model.encode(query)
            similarity = util.dot_score(query_embedding, passage_embedding)[0] # shape (1, num_paragraphs) -> (num_paragraphs,)
            similarity_matrix = similarity.reshape(1, -1)
        return similarity_matrix
   
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    return similarity_matrix


def basic_rerank(dataset, sim_method, max_num_docs, sim_threshold):
    """
    Greedily rerank the documents by removing the ones that are too similar to the ones already selected.
    """
    for i, data in tqdm(enumerate(dataset)):
        reranked_ctxs = []
        if i not in similarity_matrices:
            similarity_matrices[i] = custom_similarity([x["text"] for x in data["ctxs"]], method=sim_method)
        similarity_matrix = similarity_matrices[i]
        if not eq_1(sim_threshold):
            for i, ctx in enumerate(data["ctxs"]):
                if len(reranked_ctxs) >= max_num_docs:
                    break
                if i == 0:
                    reranked_ctxs.append({"pos": i, "content": ctx})
                else:
                    too_similar = False
                    for item in reranked_ctxs:
                        sim_score = similarity_matrix[i][item["pos"]]
                        if sim_score > sim_threshold:
                            too_similar = True
                            break
                    if not too_similar:
                        reranked_ctxs.append({"pos": i, "content": ctx})

            # get rid of pos, and only keep content
            data["ctxs"] = [x["content"] for x in reranked_ctxs]

    return dataset


def mmr_rerank(dataset, sim_method, max_num_docs, lambda_param):
    """
    Greedily rerank the documents using Maximal Marginal Relevance (MMR) algorithm.
    In each iteration, select the document that has the highest MMR score
    MMR score = lambda * similarity(query, document) - (1 - lambda) * max(similarity(document, selected))
    """
    for i, data in tqdm(enumerate(dataset)):
        reranked_ctxs = []
        ctx_texts = [x["text"] for x in data["ctxs"]]

        # update and fetch global similarity matrices
        if i not in similarity_matrices:
            similarity_matrices[i] = custom_similarity(ctx_texts, method=sim_method)
        if i not in query_similarity:
            query_similarity[i] = custom_similarity(ctx_texts, query=data["question"], method=sim_method)
        similarity_matrix = similarity_matrices[i]
        query_sim = query_similarity[i]

        # rerank the documents
        while len(reranked_ctxs) < max_num_docs:
            max_mmr_score = -1
            max_mmr_pos = -1
            for i, _ in enumerate(data["ctxs"]):
                if i in reranked_ctxs:
                    continue
                mmr_score = lambda_param * query_sim[0][i]
                if reranked_ctxs != []:
                    mmr_score -= (1 - lambda_param) * max([similarity_matrix[i][x] for x in reranked_ctxs])
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_pos = i
            reranked_ctxs.append(max_mmr_pos)

        # get rid of pos, and only keep content
        data["ctxs"] = [data["ctxs"][x] for x in reranked_ctxs]

    return dataset


def activeRDD_rerank(dataset, sim_method, max_num_docs, alpha_beta):
    """
    Greedily rerank the documents using Active RDD algorithm.
    In each iteration, select the document that has the highest Active RDD score

    relevance = similarity(query, document)
    density = sum(similarity(document, all)) / |all|
    diversity = max(similarity(document, selected))
    Active RDD score = alpha * relevance - beta * density + (1 - alpha - beta) * diversity
    """
    alpha_param, beta_param = alpha_beta
    for i, data in tqdm(enumerate(dataset)):
        reranked_ctxs = []
        ctx_texts = [x["text"] for x in data["ctxs"]]

        # update and fetch global similarity matrices
        if i not in similarity_matrices:
            similarity_matrices[i] = custom_similarity(ctx_texts, method=sim_method)
        if i not in query_similarity:
            query_similarity[i] = custom_similarity(ctx_texts, query=data["question"], method=sim_method)
        similarity_matrix = similarity_matrices[i]
        query_sim = query_similarity[i]

        # rerank the documents
        while len(reranked_ctxs) < max_num_docs:
            max_active_rdd_score = -1
            max_active_rdd_pos = -1
            for i, _ in enumerate(data["ctxs"]):
                if i in reranked_ctxs:
                    continue
                relevance = query_sim[0][i]
                density = sum(similarity_matrix[i]) / len(similarity_matrix[i])
                diversity = max([similarity_matrix[i][x] for x in reranked_ctxs]) if reranked_ctxs != [] else 0
                active_rdd_score = alpha_param * relevance - beta_param * density + (1 - alpha_param - beta_param) * diversity
                if active_rdd_score  > max_active_rdd_score:
                    max_active_rdd_score = active_rdd_score
                    max_active_rdd_pos = i
            reranked_ctxs.append(max_active_rdd_pos)

        # get rid of pos, and only keep content
        data["ctxs"] = [data["ctxs"][x] for x in reranked_ctxs]

    return dataset


def kmeans_rerank(dataset, sim_method, max_num_docs, k, max_iter=100, n_init=10):
    """
    Rerank the documents using KMeans clustering.
    Steps for each query:
    1. Cluster the documents into k clusters
    2. Sort documents in each cluster by similarity to the query
    3. Put them in the reranked list
    4. Repeat 2-3 until all documents are selected
    """
    for data in tqdm(dataset):
        reranked_ctxs = []
        ctx_texts = [x["text"] for x in data["ctxs"]]

        # Convert the documents to a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(ctx_texts)

        # Perform clustering using KMeans
        kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, random_state=0)
        kmeans.fit(X)

        similarities = custom_similarity(ctx_texts, query=data["question"], method=sim_method)
        all_cluster_doc_ids = {i: [] for i in range(k)}
        
        # For each cluster, select the documents in order of their similarity to the query
        for cluster in range(k):
            # get the indices of the documents in the cluster
            cluster_doc_ids = [i for i, label in enumerate(kmeans.labels_) if label == cluster]
            all_cluster_doc_ids[cluster] = sorted(cluster_doc_ids, key=lambda x: similarities[0][x], reverse=True)

        # Put the documents in the reranked list
        # choose 1 from the first cluster, 1 from the second cluster, and so on
        for _ in range(max_num_docs):
            for cluster in range(k):
                if all_cluster_doc_ids[cluster]:
                    # pop the first element from the clust
                    reranked_ctxs.append(all_cluster_doc_ids[cluster].pop(0))

        data["ctxs"] = [data["ctxs"][x] for x in reranked_ctxs]

    return dataset

# %%

def load_json(file_path, debug=False):
    with open(file_path, "r") as f:
        data = json.load(f)
    if debug:
        data = data[:10]
    return data


def check_output_file(output_file):
    """
    prompt the user to input yes/no if the file already exists
    """
    if os.path.exists(output_file):
        overwrite = input(f"The file {output_file} already exists. Overwrite? (yes/no): ")
        if overwrite.lower() != "yes":
            print("Exiting...")
            sys.exit(0)

def main(args):
    global model
    print_args(args)

    if args.sim_method == "sbert":
        model_name = "all-mpnet-base-v2"
        model = SentenceTransformer(model_name)

    dataset = load_json(args.input_file, args.debug)

    if args.algo == "basic":
        assert args.sim_thresholds is not None, "Please provide the similarity thresholds for basic rerank."
        param_list = args.sim_thresholds
        rerank_func = basic_rerank
    elif args.algo == "mmr":
        sim_threshold = 1 # dummy value
        param_list = args.lambda_params
        rerank_func = mmr_rerank
    elif args.algo == "kmeans":
        sim_threshold = 1 # dummy value
        param_list = args.k
        rerank_func = kmeans_rerank
    elif args.algo == "activeRDD":
        sim_threshold = 1 # dummy value
        param_list = [(alpha, beta) for alpha in args.alpha_params for beta in args.beta_params]
        rerank_func = activeRDD_rerank
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented.")
    
    for param in param_list:
        if args.algo == "basic":
            output_file = args.input_file.replace(".json", f"-reranked-{args.sim_method}-{param}.json")
        elif args.algo == "mmr":
            output_file = args.input_file.replace(".json", f"-reranked-{args.algo}-{param}-{args.sim_method}-{sim_threshold}.json")
        elif args.algo == "kmeans":
            output_file = args.input_file.replace(".json", f"-reranked-{args.algo}-{param}-{args.sim_method}-{sim_threshold}.json")
        elif args.algo == "activeRDD":
            # avoid negative weight for diversity 
            if param[0] + param[1] > 1:
                continue
            output_file = args.input_file.replace(".json", f"-reranked-{args.algo}-{param[0]}-{param[1]}-{args.sim_method}-{sim_threshold}.json")
        if args.debug:
            output_file = output_file.replace(".json", "-debug.json")
        check_output_file(output_file)

        reranked_dataset = rerank_func(dataset, args.sim_method, args.max_num_docs, param)
        with open(output_file, "w") as f:
            json.dump(reranked_dataset, f, indent=4)
        print(f"Reranked dataset saved to {output_file}")
        
        if args.debug:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_num_docs", type=int, default=100)

    # Rerank params
    parser.add_argument("--sim_method", type=str, choices=["tfidf", "spacy", "sbert"], required=True)
    parser.add_argument("--algo", type=str, choices=["basic", "mmr", "kmeans", "activeRDD"], required=True)

    # Basic rerank params
    parser.add_argument("--sim_thresholds", nargs='+', type=float) # list of similarity thresholds e.g., 0.5, 0.6, 0.7

    # MMR param
    parser.add_argument("--lambda_params", nargs='+', type=float)

    # KMeans param
    parser.add_argument("--k", nargs='+', type=int)

    # ActiveRDD param
    parser.add_argument("--alpha_params", nargs='+', type=float)
    parser.add_argument("--beta_params", nargs='+', type=float)

    args = parser.parse_args()
    main(args)