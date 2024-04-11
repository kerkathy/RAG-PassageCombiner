"""
Convert any custom dataset to the format of the topic dataset to be used in the retrieval pipeline.
Implement hotpotQA first.

topic.tsv format:
<id> <question>

Usage:
python convert_custom_to_topic.py \
    --input <path_to_input_file> \
    --output <path_to_output_file> \
    --dataset <hotpot> \
    --function <id_conversion|topic_conversion> \
    --mapping-output <path_to_mapping_output_file>
"""

import os
import csv
import json
import argparse
from tqdm import tqdm

def check_output_path(output_path):
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists")
        replace = input("Do you want to replace it? (yes/no): ")
        if replace.lower() != 'yes':
            exit()

def write_to_tsv(data, output):
    with open(output, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)
    print(f"Saved to {output}")

def write_to_json(data, output):
    with open(output, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {output}")

def convert_tsv_id_to_index(args):
    """
    Given a tsv file where each line is <id> <question>
    Convert id to 0-based index and save the mapping to a json file
    Store the mapping id->index in a dictionary
    Write to a json file
    """
    # make sure input is a tsv file
    if not args.input.endswith(".tsv"):
        raise ValueError("Input file must be a tsv file")
    
    # csv reader
    with open(args.input, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)

    mapping = {}
    for i, row in enumerate(data):
        # i is the index
        # row[0] is the id
        mapping[row[0]] = i
        # convert id to index
        row[0] = i

    # Write to file
    write_to_json(mapping, args.mapping_output)
    write_to_tsv(data, args.output)

def convert_hotpot_to_topic(args):
    # Given a hotpotQA json file, convert to the topic format: <id>\t<question>
    # Also save the mapping from id to index
    """
    Given a hotpotQA json file, convert to the topic tsv
    Target format: <question>\t<answer>
    """
    # make sure input is a json file
    if not args.input.endswith(".json"):
        raise ValueError("Input file must be a json file")
    
    # Load the dataset
    with open(args.input, "r") as f:
        data = json.load(f)

    # Convert to topic format
    topic_data = []
    mapping = {}
    if args.format == "qa":
        for example in tqdm(data):
            if type(example["answer"]) != list:
                example["answer"] = [example["answer"]]
            topic_data.append([example["question"], example["answer"]])
    elif args.format == "str-id":
        for i, example in tqdm(enumerate(data)):
            topic_data.append([example["_id"], example["question"]])
    elif args.format == "int-id":
        for i, example in tqdm(enumerate(data)):
            topic_data.append([i, example["question"]])
            mapping[example["_id"]] = i
    else:
        raise ValueError(f"args.format {args.format} not supported")

    # Write to file
    write_to_tsv(topic_data, args.output)
    # if args.format == "int-id":
    #     write_to_json(mapping, args.mapping_output)



def main(args):
    check_output_path(args.output)

    if args.function == "id_conversion":
        convert_tsv_id_to_index(args)
    elif args.function == "topic_conversion":
        convert_hotpot_to_topic(args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["hotpot", "msmarcoqa", "msmarco-v2-subset", "eli5", "strategyQA", "AmbigQA"], help="Dataset to convert")
    parser.add_argument("-f", "--function", type=str, required=True, choices=["id_conversion", "topic_conversion"], help="Function to run")
    parser.add_argument("-m", "--mapping-output", type=str, help="Path to the mapping file")
    parser.add_argument("--format", type=str, default="qa", choices=["qa", "int-id", "str-id"], help="Format for topic data")
    args = parser.parse_args()
    main(args)