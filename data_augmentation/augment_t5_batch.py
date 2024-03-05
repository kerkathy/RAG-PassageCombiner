"""
Paraphrase each data for once.
Name the output file that contains a paraphrase as xxxx-1.tsv, xxxx-2.tsv, xxxx-3.tsv,
Usage: `python augment_t5_batch.py --split <split> --device <cuda> --seed <seed> --paraphrase <paraphrase>`
"""

# %%
import sys
import torch
import argparse
import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

# %%
def main():
    # %%
    # check number of arguments and print help
    if len(sys.argv) != 9:
        print("Usage: python augment_t5_batch.py --split <split> --device <cuda> --seed <seed> --paraphrase <paraphrase>")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='For different splits.')
    parser.add_argument('--split', type=str, default="1", help='Split number')
    parser.add_argument('--device', type=int, default=0, help='Cuda number')
    parser.add_argument('--seed', type=int, default=1, help='Seed number')
    parser.add_argument('--paraphrase', type=int, default=1, help='Which round of paraphrase')
    args = parser.parse_args()

    # shuold be in the range of 1-40
    valid_split_numbers = [str(i) for i in range(1, 41)]
    if args.split not in valid_split_numbers:
        raise Exception("Please enter a valid split number.")

    input_file_name = f"../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100-{args.split}.tsv"
    if args.paraphrase != 0:
        output_file_name = input_file_name.replace(f".tsv", f"-paraphrase-{args.paraphrase}.tsv")
    else:
        raise Exception("Please enter a valid paraphrase number (>0).")
    
    # %%
    raw_dataset = load_dataset("csv", data_files=input_file_name, delimiter="\t")

    # Special case: replace NaN with "NaN" 
    # (that's a title of a wikipedia page, but wrongly interpreted as NaN)
    def replace_nan(example):
        for key, value in example.items():
            if pd.isnull(value):
                example[key] = "NaN"
        return example

    raw_dataset = raw_dataset.map(replace_nan)

    raw_dataset = raw_dataset["train"] # now we only have train dataset

    # %%
    batch_size = 180
    max_length = 256
    max_sample = 10 # number of samples to be augmented
    top_k = 100
    top_p = 0.95
    set_seed(args.seed)

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # %%
    model_name_or_path = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
    max_seq_length = min(max_length, tokenizer.model_max_length)
    print(f"Tried to set max_seq_length to {max_length}, but was set to {max_seq_length}")

    # %%
    rows_list = []
    time_list = []

    # Iterate over batches
    dataloader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader):
        start_time = time.time()
        batch["prompt"] = ["paraphrase: " + text + " </s>" for text in batch["text"]]
        tokenized_inputs = tokenizer(
            batch["prompt"], 
            padding="max_length", max_length=max_seq_length, truncation=True,
            return_tensors="pt"
        )
        tokenized_inputs = tokenized_inputs.to(device)

        # print len of tokens that are not padding
        # print(tokenized_inputs.input_ids.ne(tokenizer.pad_token_id).sum(dim=1))
        
        # Generate paraphrases
        with torch.no_grad():
            paraphrase_ids = model.generate(
                input_ids=tokenized_inputs.input_ids, attention_mask=tokenized_inputs.attention_mask, 
                max_length=max_seq_length,
                do_sample=True,
                top_k=top_k, 
                top_p=top_p,
                early_stopping=False,
                num_return_sequences=1,
            )
        
        # Decode the paraphrases and add them to the batch
        paraphrases = tokenizer.batch_decode(paraphrase_ids, skip_special_tokens=True)

        # Iterate over the paraphrases and create a new row for each one
        for i, paraphrase in enumerate(paraphrases):
            new_row = (batch["id"][i].item(), paraphrase, batch["title"][i])
            rows_list.append(new_row)
        end_time = time.time()
        time_list.append(end_time - start_time)

    # %%
    df_paraphrases = pd.DataFrame(rows_list, columns=["para_id", "text", "title"])
    df_paraphrases.to_csv(output_file_name, sep='\t', index=False)

    # %%
    print("Average time taken in seconds: ", sum(time_list)/len(time_list))

# %%
if __name__ == '__main__':
    main()
