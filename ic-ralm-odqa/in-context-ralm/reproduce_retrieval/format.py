"""
Usage: python format.py --input_file /path/to/input.json --output_file /path/to/output.json \
    [--have_raw] [--impact] [--index_name /path/to/index] [--device 0] [--query_encoder /path/to/encoder]
    
Given a structure like this:
{
    "0": {
        "question": "who got the first nobel prize in physics",
        "answers": [
            "Wilhelm Conrad R\u00f6ntgen"
        ],
        "contexts": [
            {
                "docid": "284453",
                "score": "82.007050",
                "has_answer": false
            },...
        ]
    },...
}

Convert it to this:
[
    {
        "question": "who got the first nobel prize in physics",
        "answers": [
            "Wilhelm Conrad R\u00f6ntgen"
        ],
        "ctxs": [
            {
                "id": "284453",
                "title": "Nobel Prize",
                "text": "A group including 42 Swedish writers, artists, ... The first Nobel Peace Prize went to the Swiss",
                "score": "82.00703",
                "has_answer": false
            },...
        ]
    },...
]
"""

# %%
import argparse
import json
import os
from collections import OrderedDict

from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
from tqdm import tqdm
# %%

def replace_keys(d, old_key, new_key):
    d[new_key] = d.pop(old_key)

def add_key(d, key):
    d[key] = None

def order_keys(d, keys):
    d = OrderedDict(sorted(d.items(), key=lambda x: keys.index(x[0])))
    return d

def get_raw(args, dataset):
    if args.impact:
        if os.path.exists(args.luc_index_name):
            searcher = LuceneImpactSearcher(args.luc_index_name, args.query_encoder)
        else:
            searcher = LuceneImpactSearcher.from_prebuilt_index(args.luc_index_name, args.query_encoder)
    else:
        if os.path.exists(args.luc_index_name):
            searcher = LuceneSearcher(args.luc_index_name)
        else:
            searcher = LuceneSearcher.from_prebuilt_index(args.luc_index_name)

    # Iterate over the dataset and perform retrieval
    for example in tqdm(dataset):
        # Add the hits to the example
        if args.have_title:
            for ctx in example["ctxs"]:
                contents = searcher.doc(int(ctx["id"])).contents()
                if contents == None:
                    contents = searcher.doc(int(ctx["id"])).raw()
                contents = json.loads(contents)["contents"]
                ctx["title"] = contents.split("\n")[0]
                ctx["text"] = "\n".join(contents.split("\n")[1:])
        else:
            for ctx in example["ctxs"]:
                contents = searcher.doc(int(ctx["id"])).contents()
                if contents == None:
                    contents = searcher.doc(int(ctx["id"])).raw()
                contents = json.loads(contents)["contents"]
                ctx["text"] = contents

# %%
def main(args):
    # check if the output file already exists
    # ask if we should overwrite it
    if os.path.exists(args.output_file):
        overwrite = input(f"The file {args.output_file} already exists. Overwrite? (yes/no): ")
        if overwrite.lower() != "yes":
            print("Exiting...")
            return
        
    if args.add_id and not args.id_dataset:
        raise ValueError("Need to provide id_dataset if add_id is True")

    # Load the dataset
    with open(args.input_file, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples from {args.input_file}")

    # dataset is a dict
    # Replace "contexts" with "ctxs"
    for example in dataset.values():
        replace_keys(example, "contexts", "ctxs")
        for ctx in example["ctxs"]:
            replace_keys(ctx, "docid", "id")

    if not args.have_title:
        for example in dataset.values():
            for ctx in example["ctxs"]:
                add_key(ctx, "title")
                
    for example in dataset.values():
        for i, ctx in enumerate(example["ctxs"]):
            example["ctxs"][i] = order_keys(ctx, ["id", "title", "text", "score", "has_answer"])
    
    if args.add_id:
        with open(args.id_dataset, "r") as f:
            id_dataset = json.load(f) # list of dict
            ids = [d["_id"] for d in id_dataset]
        
        # Add the id to the dataset
        for i, example in enumerate(dataset.values()):
            example["_id"] = ids[i]

    # Fetch raw text if needed
    if not args.have_raw:
        get_raw(args, dataset)

    # Save the dataset
    with open(args.output_file, "w") as f:
        # convert dict to list
        json.dump(list(dataset.values()), f, indent=4)

    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--have_title", action="store_true") # whether contents are title\ntext or just text
    parser.add_argument("--have_raw", action="store_true")
    parser.add_argument("--index_name", type=str) # no need if have_raw
    parser.add_argument("--impact", action="store_true", help="Use Impact index")
    parser.add_argument("--query_encoder", type=str, help="Needed for Impact Lucene Index")
    parser.add_argument("--add_id", action="store_true") # for msmarcoqa dataset
    parser.add_argument("--id_dataset", type=str) # for msmarcoqa dataset

    args = parser.parse_args()

    main(args)