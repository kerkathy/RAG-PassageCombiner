# I have a tsv with the columns: id, text, title
# I want to check if a target string is in any text or not

import pandas as pd

def check_existence(df, target):
    """
    Check if a target string is in any text or not
    :param df: pd.DataFrame
    :param target: str
    :return: pd.DataFrame
    """
    df['text'] = df['text'].apply(lambda x: target in x)

    print("Num of existence in text:", df['text'].sum())


if __name__ == '__main__':
    tsv_file = "../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100.tsv"
    print("Reading tsv file...")
    df = pd.read_csv(tsv_file, sep='\t', nrows=100)
    print("Done!")
    print("")

    print("Shape:", df.shape) # (21015324, 3)
    print("Columns:", df.columns) # Index(['id', 'text', 'title'], dtype='object')

    # extract first 30 rows
    df = df.iloc[:30, :]
    # save to tsv
    df.to_csv('../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100-30.tsv', sep='\t', index=False)
    print("Saved to ../DPR/dpr/resource/downloads/data/wikipedia_split/psgs_w100-30.tsv")

    # target_string = "There are locations across the United States as well as in other countries such as Canada, Puerto Rico"
    # print("Checking if the target string is in any text...")
    # check_existence(df, target_string)
    print("Done!")