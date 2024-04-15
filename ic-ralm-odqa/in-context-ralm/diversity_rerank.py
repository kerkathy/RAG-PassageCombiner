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
        similarity_scores: list of lists, where similarity_scores[i][j] is the similarity score between paragraph i and j
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


def basic_rerank(dataset, sim_method, sim_threshold=0.5):
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


def mmr_rerank(dataset, sim_method, lambda_param):
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
        while len(reranked_ctxs) < len(data["ctxs"]):
            max_mmr_score = -1
            max_mmr_pos = -1
            for i, ctx in enumerate(data["ctxs"]):
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
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not implemented.")
    
    for param in param_list:
        if args.algo == "basic":
            output_file = args.input_file.replace(".json", f"-reranked-{args.sim_method}-{param}.json")
        elif args.algo == "mmr":
            output_file = args.input_file.replace(".json", f"-reranked-{args.algo}-{param}-{args.sim_method}-{sim_threshold}.json")
        if args.debug:
            output_file = output_file.replace(".json", "-debug.json")
        check_output_file(output_file)

        reranked_dataset = rerank_func(dataset, args.sim_method, param)
        with open(output_file, "w") as f:
            json.dump(reranked_dataset, f, indent=4)
        print(f"Reranked dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    # Rerank params
    parser.add_argument("--sim_method", type=str, choices=["tfidf", "spacy", "sbert"], required=True)
    parser.add_argument("--algo", type=str, choices=["basic", "mmr"], required=True)

    # Basic rerank params
    parser.add_argument("--sim_thresholds", nargs='+', type=float) # list of similarity thresholds e.g., 0.5, 0.6, 0.7

    # MMR param
    parser.add_argument("--lambda_params", nargs='+', type=float)

    args = parser.parse_args()
    main(args)