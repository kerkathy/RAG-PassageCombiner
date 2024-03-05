"""
convert a tsv file to jsonl file
para_id -> id (current_max_idx + para_id)
text, title -> contents (<title>/n<text>)
Usage: python tsv_to_jsonl.py --split 26 --paraphrase 0 [--subset]
"""
# %%
import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm


def main():
    # 3-6 should be indicated in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="26", required=True)
    parser.add_argument("--paraphrase", type=int, default=0) # 0 means orig, 1 means para, 2 means second para
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args()

    source_tsv_file = f"../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100-{args.split}.tsv"
    target_jsonl_file = f"../DPR/dpr/resource/downloads/data/wiki_split_orig_jsonl/psgs_w100-{args.split}.jsonl"
    if args.paraphrase != 0:
        source_tsv_file = source_tsv_file.replace(".tsv", f"-paraphrase-{args.paraphrase}.tsv")
        target_jsonl_file = target_jsonl_file.replace(".jsonl", f"-paraphrase-{args.paraphrase}.jsonl")
        target_jsonl_file = target_jsonl_file.replace("wiki_split_orig_jsonl", f"wiki_split_para_{args.paraphrase}_jsonl")
    if args.subset:
        source_tsv_file = source_tsv_file.replace(".tsv", "-subset.tsv")
        source_tsv_file = source_tsv_file.replace("wikipedia_split", "wikipedia_split_subset")
        target_jsonl_file = target_jsonl_file.replace(".jsonl", "-subset.jsonl")
    
    current_max_idx = 21015324 * args.paraphrase

    # Check if target file already exists
    try:
        with open(target_jsonl_file, "r") as f:
            print("File already exists:", target_jsonl_file)
            return
    except FileNotFoundError:
        pass

    os.makedirs(os.path.dirname(target_jsonl_file), exist_ok=True)

    # %%
    # measure time
    start_time = time.time()
    print("Reading file", source_tsv_file)
    df = pd.read_csv(source_tsv_file, sep="\t")
    print("df.shape:", df.shape)
    print("Time taken:", time.time() - start_time)

    # %%
    print("Writing to", target_jsonl_file)
    start_time = time.time()
    with open(target_jsonl_file, "w") as f:
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            contents = str(row["title"]) + "\n" + str(row["text"])
            if args.paraphrase:
                new_id = current_max_idx + row["para_id"]
            else:
                new_id = row["id"]
            json.dump({"id": str(new_id), "contents": contents}, f)
            f.write("\n")

    print("Written to", target_jsonl_file)
    print("Time taken:", time.time() - start_time)

# %%
if __name__ == "__main__":
    main()