"""
Sanity check for the paraphrase dataset.
Compute the similarity between paraphrased and original sentences 
v.s. the similarity between paraphrased and random sentences. 
"""

# %%
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# %%
def load_tsv_dataset(file_name, col_to_remove=None):
    dataset = load_dataset("csv", data_files=file_name, delimiter="\t") # (id or para_id), text, title
    dataset = dataset["train"] # now we only have train dataset
    if col_to_remove:
        dataset = dataset.remove_columns(col_to_remove)
    return dataset

# Compute DPR embeddings
def compute_embeddings(dataset):
    embeddings = []
    for i in tqdm(range(len(dataset))):
        embeddings.append(model.encode(dataset[i]["text"]))
    return embeddings

# Write stats to file
def write_stats(similarity_scores, file_name):
    similarity_scores = np.array(similarity_scores)
    mean = np.mean(similarity_scores)
    std = np.std(similarity_scores)
    highest = np.max(similarity_scores)
    lowest = np.min(similarity_scores)
    with open(file_name, "w") as f:
        f.write(f"mean: {mean}\n")
        f.write(f"std: {std}\n")
        f.write(f"highest: {highest}\n")
        f.write(f"lowest: {lowest}\n")
        for score in similarity_scores:
            f.write(str(score) + "\n")

# %%
original_file = "../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100-30.tsv"
paraphrase_file = "../data_augmentation/archive/psgs_w100-30-paraphrase.tsv"

original_dataset = load_tsv_dataset(original_file, col_to_remove="title")
paraphrase_dataset = load_tsv_dataset(paraphrase_file, col_to_remove="title")
# %%
# use DPR to encode the sentences (to align with retriever we use)
model = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
# %%
original_dataset = original_dataset.add_column("embeddings", compute_embeddings(original_dataset))
paraphrase_dataset = paraphrase_dataset.add_column("embeddings", compute_embeddings(paraphrase_dataset))

# %%
original_dataset = original_dataset.shuffle(seed=42)
print(original_dataset.select(range(5)))

# %%
# compute similarity between paraphrase and original, and between random and original
similarity_paraphrase_original = []
similarity_random_original = []
for i in tqdm(range(len(original_dataset))):
    paraphrase_sentence = paraphrase_dataset.filter(lambda x: x["para_id"] == original_dataset[i]["id"])
    similarity_paraphrase_original.append(util.pytorch_cos_sim(original_dataset[i]["embeddings"], paraphrase_sentence["embeddings"])[0][0].item())

    random_sentence = original_dataset[i+1] if i+1 < len(original_dataset) else original_dataset[0]
    similarity_random_original.append(util.pytorch_cos_sim(original_dataset[i]["embeddings"], random_sentence["embeddings"])[0][0].item())

# %%
write_stats(similarity_paraphrase_original, "similarity_paraphrase_original.txt")
write_stats(similarity_random_original, "similarity_random_original.txt")
# %%
