# """
# Given a json file of a list of dict containing a key "ctxs"
# the value of "ctxs" is a list of dict
# add a key "title" to each dict in the list
# where the value of "title" is None

# Usage:
# python tmp.py --input /path/to/input.json --output /path/to/output.json --key title
# """

# import json
# import os
# import argparse
# from tqdm import tqdm


# def add_key(d, key):
#     d[key] = None
    
# def reverse_dict(d):
#     return {v: k for k, v in d.items()}


# def main(args):
#     # check if the output file already exists
#     if os.path.exists(args.output):
#         print(f"Output file {args.output} already exists")
#         replace = input("Do you want to replace it? (yes/no): ")
#         if replace.lower() != 'yes':
#             return

#     # load the input file
#     with open(args.input, "r") as f:
#         data = json.load(f)

#     if args.debug:
#         if type(data) == list:
#             data = data[:5]
#         elif type(data) == dict:
#             # keep only the first 5 keys
#             keys = list(data.keys())[:5]
#             data = {k: data[k] for k in keys}

#     # for example in tqdm(data):
#     #     for ctx in example["ctxs"]:
#     #         add_key(ctx, args.key)

#     # write the output file
#     with open(args.output, "w") as f:
#         json.dump(reverse_dict(data), f, indent=4)
#         # json.dump(data, f, indent=4)

#     print(f"Saved to {args.output}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", type=str, required=True, help="input json file")
#     parser.add_argument("-o", "--output", type=str, required=True, help="output json file")
#     parser.add_argument("-k", "--key", type=str, help="key to add to each dict in the list")
#     parser.add_argument("-d", "--debug", action="store_true")
#     # parser.add_argument("--input", type=str, required=True, help="input json file")
#     # parser.add_argument("--output", type=str, required=True, help="output json file")
#     # parser.add_argument("--key", type=str, required=True, help="key to add to each dict in the list")
#     # parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()
#     main(args)

# %%
import json
import random
# read a json file and see the number of keys in each dict

# input_path = "result/dpr-trivia-test/formatted-ms2.dpr-trivia-test.hits-100.json" # list, 11313
# input_path = "result/nq-test/formatted-ms2.nq-test.hits-100.json" # list, len=3610
# input_path = "result/hotpot/formatted-ms2.hotpot.hits-100.json" # list, len=7405
input_path = "/home/guest/r11944026/research/ic-ralm-odqa/in-context-ralm/data/msmarco-qa/MSMARCO-Question-Answering/Data/train_v2.1_nlgen.json"

with open(input_path, "r") as f:
    data = json.load(f)

if type(data) == list:
    print("TYpe of data:", type(data))
    print("Length of data:", len(data))
    print("Type of data[0]:", type(data[0]))
elif type(data) == dict:
    print("Type of data:", type(data))
    # print("Length of data[0]:", len(data[list(data.keys())[0]]))
    # print("Length of data[1]:", len(data[list(data.keys())[1]]))
    # print("Length of data[4]:", len(data[list(data.keys())[4]]))
    print("Keys:", list(data.keys()))

    # %%
    # data is a dict of dict of list
    # print the first k items of each inner dict
    # e.g., data["passages"] is a dict
    # print the first k keys of data["passages"]
    # do so for other keys of data

    # convert to a list of dict
    # with keys _id, question, answer
    k = 12467
    seed = 42
    random.seed(seed)

    # random sample k ids from the list of keys
    all_ids = list(data["query"].keys())
    random.shuffle(all_ids)
    k_ids = all_ids[:k]

    output_list = []

    for _id in k_ids:

        output_list.append({
            "_id": _id,
            "question": data["query"][_id],
            "query_type": data["query_type"][_id],
            "answer": data["answers"][_id]
        })

    # write output_list to a json file
    output_path = input_path.replace(".json", f"-subset-{k}.json")
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=4)

    print(f"Saved to {output_path}")
# %%