"""
Given a list of dictionaries, where each dict has keys "question", "answers", and "ctxs"
Rerank the ctxs based on diversity.
"""

import os
import json
import argparse
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrices = {}

def eq_1(x):
    epsilon = 1e-9  # a very small number
    if abs(x - 1) < epsilon:
        return True
    else:
        return False


def custom_similarity(paragraphs, method="tfidf"):
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
        score_matrix = vectorizer.fit_transform(paragraphs)
        similarity_matrix = cosine_similarity(score_matrix)

    elif method == "spacy":
        nlp = spacy.load("en_core_web_lg")
        docs = [nlp(text) for text in paragraphs]
        similarity_matrix = [[doc1.similarity(doc2) for doc2 in docs] for doc1 in docs]
    
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    return similarity_matrix


# def diversity_rerank(dataset, sim_method="tfidf", sim_threshold=0.5):
#     for data in tqdm(dataset):
#         reranked_ctxs = []
#         similarity_matrix = custom_similarity([x["text"] for x in data["ctxs"]], method=sim_method)
#         if not eq_1(sim_threshold):
#             for i, ctx in enumerate(data["ctxs"]):
#                 if i == 0:
#                     reranked_ctxs.append({"pos": i, "content": ctx})
#                 else:
#                     too_similar = False
#                     for item in reranked_ctxs:
#                         sim_score = similarity_matrix[i][item["pos"]]
#                         if sim_score > sim_threshold:
#                             too_similar = True
#                             break
#                     if not too_similar:
#                         reranked_ctxs.append({"pos": i, "content": ctx})
#                         # if len(reranked_ctxs) == num_docs:
#                         #     break

#             # get rid of pos, and only keep content
#             data["ctxs"] = [x["content"] for x in reranked_ctxs]

#     return dataset


def diversity_rerank(dataset, sim_method="tfidf", sim_threshold=0.5):
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


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def main(args):
    dataset = load_json(args.input_file)

    for sim_threshold in args.sim_thresholds:
        output_file = args.input_file.replace(".json", f"-reranked-{args.sim_method}-{sim_threshold}.json")

        # prompt the user to input yes/no if the file already exists
        if os.path.exists(output_file):
            overwrite = input(f"The file {output_file} already exists. Overwrite? (yes/no): ")
            if overwrite.lower() != "yes":
                print("Exiting...")
                return

        # Use the precalculated similarity matrix
        reranked_dataset = diversity_rerank(dataset, args.sim_method, sim_threshold)

        with open(output_file, "w") as f:
            json.dump(reranked_dataset, f, indent=4)

        print(f"Reranked dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)

    # Rerank params
    parser.add_argument("--sim_method", type=str, choices=["tfidf", "spacy"], required=True)
    parser.add_argument("--sim_thresholds", nargs='+', type=float, required=True)

    args = parser.parse_args()
    main(args)