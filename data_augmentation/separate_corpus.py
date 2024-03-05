# Separate the corpus into 4 splits
# Each split contains 1/4 of the corpus
# %%
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='For different splits.')
parser.add_argument('SplitNumber', metavar='split', type=int, help='the split number to be added to the input file name')
args = parser.parse_args()

input_file_name = f"../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100-{args.SplitNumber}.tsv"
df = pd.read_csv(input_file_name, sep="\t")

# %%
num_splits = 10
split_size = int(len(df) / num_splits)
new_header = df.columns.values
df = df[0:] # remove the header

print(f"split_size: {split_size}")

for i in range(num_splits):
    if i == num_splits - 1:
        df_split = df[i*split_size:]
    else:
        df_split = df[i*split_size:(i+1)*split_size]
    print(f"{i}th df_split.shape: {df_split.shape}")

    output_file_name = input_file_name.replace(".tsv", f"-{i+1}.tsv")
    df_split.to_csv(output_file_name, sep="\t", header=new_header, index=False)
    print("Written to file.")

# %%
