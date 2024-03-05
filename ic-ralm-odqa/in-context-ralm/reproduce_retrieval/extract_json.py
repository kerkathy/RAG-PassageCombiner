# Extract the first portion of a json file
# Usage: python extract_json.py <size_of_subset> <input_file> <output_file>
# Example: python extract_json.py 5 data/ralm/ralm_train.json data/ralm/ralm_train_1.json

import json
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_json.py <size_of_subset> <input_file> <output_file>")
        sys.exit(1)

    output_size = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Check if data is a list and then slice it
    if isinstance(data, list):
        print(f"There are {len(data)} items in the list.")
        data = data[:output_size]
    else:
        print("JSON data is not a list and cannot be sliced.")
        print(f"The first {output_size} Keys: {list(data.keys())[0:output_size]}")
        print("Try using numbers as keys to slice the data.")
        indices = range(output_size)
        data = {str(i): data[str(i)] for i in indices}

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
        print(f"JSON data written to {output_file}")

if __name__ == '__main__':
    main()