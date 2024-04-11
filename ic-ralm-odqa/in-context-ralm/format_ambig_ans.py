"""
Format a jsonl file (orig_ans) where each line is like: 
{"query_id": 1, "answers": [" none"]}

Into a json file (new_ans) containing a dict with the key being the converted query_id:
{
  "-6631842452804060768": [" none"],
  ...
}

And write a id_conversion function that converts the query_id to the key
The mapping is indicated from a json file (orig_ref) containing a list of dict
"""

import argparse
import json

def get_orig_ids(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
        ids = [d["id"] for d in data]

    return ids

def format_ambig_ans(orig_ans, orig_ref, new_ans):
    if not orig_ans.endswith(".json"):
        raise ValueError("orig_ans must be a json file")
    if not orig_ref.endswith(".json"):
        raise ValueError("orig_ref must be a json file")
    if not new_ans.endswith(".json"):
        raise ValueError("new_ans must be a json file")
    
    orig_ids = get_orig_ids(orig_ref)
    
    new_ans_data = {}
    with open(new_ans, "w") as g:
        with open(orig_ans, "r") as f:
            for line in f:
                data = json.loads(line)
                query_id = data["query_id"]
                new_id = orig_ids[query_id-1]
                new_ans_data[new_id] = data["answers"]
        json.dump(new_ans_data, g, indent=2)
    
    print(f"Converted {orig_ans} to {new_ans}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--new', type=str, required=True)
    args = parser.parse_args()

    format_ambig_ans(args.old, args.ref, args.new)