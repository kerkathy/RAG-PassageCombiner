# %%
## built-in
import json,os
import types
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud
os.environ["CUDA_LAUNCH_BLOCKING"]="1" ## for debugging

## third-party
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import DistributedDataParallelKwargs
import transformers
from transformers import (
    BertTokenizer,
    BertModel,
    DPRContextEncoder, 
    DPRContextEncoderTokenizer
)
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

## own
from utils import (
    get_yaml_file,
    set_seed,
    make_index,
)

from utils import normalize_query, get_positive_docid

def read_data(file):
    print(f"...Loading data from {file}...")
    data = json.load(open(file))
    print(f"Size of train data: {len(data)}")
    return data

def create_corpus(data):
    print("...Creating Corpus...")
    corpus = [[x['text'] for x in sample['ctxs']] for sample in data]
    print(f"Size of corpus: {len(corpus)}")
    return corpus

def save_if_not_exists(data, var_name, path):
    if os.path.exists(path):
        print(f"File {path} already exists. Not overwriting.")
    else:
        if path.endswith(".pt"):
            torch.save(data, path)
        elif path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        print(f"Saved the {var_name} to {path}")

class Index:
    def __init__(self, args):
        self.args = args
        self.train_doc_embeddings = None
        self.dev_doc_embeddings = None
        self.empty_doc_embedding = None
        self.train_doc_embeddings = None
        self.train_data = None

    def create(self, train=False, dev=False, empty=False):
        """
        Choose to create the index for train or dev data or both
        """
        if not train and not dev and not empty:
            return
        if "dpr" in self.args.encoder_type:
            print("Using DPR model for document encoder")
            ret_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.args.retriever_model, cache_dir=self.args.cache_dir)
            doc_encoder = DPRContextEncoder.from_pretrained(self.args.retriever_model, cache_dir=self.args.cache_dir)
        else:
            print("Using BERT model for document encoder")
            ret_tokenizer = BertTokenizer.from_pretrained(self.args.retriever_model, cache_dir=self.args.cache_dir)
            doc_encoder = BertModel.from_pretrained(self.args.retriever_model, add_pooling_layer=False, cache_dir=self.args.cache_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        doc_encoder.to(device)
        doc_encoder.eval()

        if train:
            train_data = self.read_data(self.args.train_file)
            train_corpus = create_corpus(train_data)
            self.train_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(train_corpus)]
            print(f"Train doc embedding calculated")
        if dev:
            dev_data = self.read_data(self.args.dev_file)
            dev_corpus = create_corpus(dev_data)
            self.dev_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
            print(f"DEV Index calculated")
        if empty:
            self.empty_doc_embedding = make_index(["[UNK]"], ret_tokenizer, doc_encoder).squeeze()
            print(f"Empty doc embedding calculated")

    def read_index(self, train=False, dev=False, empty=False):
        if train:
            print(f"...Loading index from {self.args.train_index_path}...")
            self.train_doc_embeddings = torch.load(self.args.train_index_path)
            print(f"Finish! Size of train index: {len(self.train_doc_embeddings)}")
        if dev:
            print(f"...Loading index from {self.args.dev_index_path}...")
            self.dev_doc_embeddings = torch.load(self.args.dev_index_path)
            print(f"Finish! Size of dev index: {len(self.dev_doc_embeddings)}")
        if empty:
            print(f"...Loading index from {self.args.empty_index_path}...")
            self.empty_doc_embedding = torch.load(self.args.empty_index_path)
            print(f"Finish! Size of empty index: {len(self.empty_doc_embedding)}")

    def extract(self):
        if self.train_doc_embeddings is not None:
            print(f"Extracting the first {self.args.train_k} train embeddings...")
            self.train_doc_embeddings = self.train_doc_embeddings[:self.args.train_k]
            self.args.train_index_path = self.args.train_index_path.replace(".pt", f"_{self.args.train_k}.pt")
            print(f"New Train index path: {self.args.train_index_path}")

        if self.dev_doc_embeddings is not None:
            print(f"Extracting the first {self.args.dev_k} dev embeddings...")
            self.dev_doc_embeddings = self.dev_doc_embeddings[:self.args.dev_k]
            self.args.dev_index_path = self.args.dev_index_path.replace(".pt", f"_{self.args.dev_k}.pt")
            print(f"New Dev index path: {self.args.dev_index_path}")

    def normalize(self):
        if self.train_doc_embeddings is not None:
            print(f"Converting train embeddings into unit vectors...")
            self.train_doc_embeddings = [F.normalize(embedding, p=2, dim=1) for embedding in tqdm(self.train_doc_embeddings)]
            self.args.train_index_path = self.args.train_index_path.replace(".pt", "_norm.pt")
            print(f"New Train index path: {self.args.train_index_path}")
        
        if self.dev_doc_embeddings is not None:
            print(f"Converting dev embeddings into unit vectors...")
            self.dev_doc_embeddings = [F.normalize(embedding, p=2, dim=1) for embedding in tqdm(self.dev_doc_embeddings)]
            self.args.dev_index_path = self.args.dev_index_path.replace(".pt", "_norm.pt")
            print(f"New Dev index path: {self.args.dev_index_path}")
        
        if self.empty_doc_embedding is not None:
            print(f"Converting empty embeddings into unit vectors...")
            self.empty_doc_embedding = F.normalize(self.empty_doc_embedding.squeeze(), p=2, dim=0)
            self.args.empty_index_path = self.args.empty_index_path.replace(".pt", "_norm.pt")
            print(f"New Empty index path: {self.args.empty_index_path}")
    
    def save_all(self):
        save_if_not_exists(self.train_doc_embeddings, "train_doc_embeddings", self.args.train_index_path)
        save_if_not_exists(self.dev_doc_embeddings, "dev_doc_embeddings", self.args.dev_index_path)
        save_if_not_exists(self.empty_doc_embedding, "empty_doc_embedding", self.args.empty_index_path)
        save_if_not_exists(self.train_data, "train_data", self.args.train_file)

    def process_and_save(self):
        action = {}
        if self.args.on_train:
            action["train"] = "read" if os.path.exists(self.args.train_index_path) else "create"
        if self.args.on_dev:
            action["dev"] = "read" if os.path.exists(self.args.dev_index_path) else "create"
        if self.args.on_empty:
            action["empty"] = "read" if os.path.exists(self.args.empty_index_path) else "create"
        to_read = [k for k,v in action.items() if v == "read"]
        to_create = [k for k,v in action.items() if v == "create"]
        self.create(train=True if "train" in to_create else False, dev=True if "dev" in to_create else False, empty=True if "empty" in to_create else False)
        self.read_index(train=True if "train" in to_read else False, dev=True if "dev" in to_read else False, empty=True if "empty" in to_read else False)
        if self.args.extract:
            self.extract()
        if self.args.normalize:
            self.normalize()
        self.save_all()

    def keep_has_positive_data(self):
        # read data
        train_qa_pairs = [(normalize_query(sample['question']), [-1], [sample['answers'][0]]) for sample in self.train_data]
        train_corpus = create_corpus(self.train_data)

        # get positive doc ids for each question and save result
        train_all_pos_doc_ids = [get_positive_docid(qa_pair[-1][0], corpus) for qa_pair, corpus in zip(train_qa_pairs, train_corpus)]
        # TODO consider all answers, and sort the answers by how many times they appear in the data
        has_postivie_qids = [i for i, docids in enumerate(train_all_pos_doc_ids) if len(docids) > 0]

        # filter out the data that has no positive doc
        self.train_doc_embeddings = [self.train_doc_embeddings[i] for i in has_postivie_qids]
        self.train_data = [
            {
                "question": data["question"],
                "answers": data["answers"],
                "ctxs": data["ctxs"],
                "all_pos_doc_ids": train_all_pos_doc_ids[i]
            }
            for i, data in enumerate(self.train_data) if i in has_postivie_qids
        ]
        print(f"[Filtered positive docs] len(train_qa_pairs): {len(train_qa_pairs)}")
        print(f"[Filtered positive docs] len(train_corpus): {len(train_corpus)}")
        print(f"[Filtered positive docs] len(self.train_doc_embeddings): {len(self.train_doc_embeddings)}")
        print(f"[Filtered positive docs] len(train_all_pos_doc_ids): {len(train_all_pos_doc_ids)}")

        self.args.train_index_path = self.args.train_index_path.replace(".pt", "_filtered_positive_docs.pt")
        self.args.train_file = self.args.train_file.replace(".json", "_filtered_positive_docs.json")
        self.save_all()

    def process_keep_has_positive_data(self):
        self.train_data = read_data(self.args.train_file)
        self.read_index(train=True, dev=False, empty=False)
        self.keep_has_positive_data()

if __name__ == '__main__':
    config_file = 'config/preprocess_idx.yaml'
    yaml_config = get_yaml_file(config_file)
    args_dict = {}
    args_dict['config_file'] = config_file

    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    set_seed(args.seed)

    index = Index(args)
    if args.keep_has_positive_data:
        if not args.on_train or args.on_dev or args.on_empty or args.extract or args.normalize:
            raise ValueError("Keeping positive data is only done on training data. Please set on_train to True; on_dev, on_empty, extract and normalize to False")
        index.process_keep_has_positive_data()
    else:
        index.process_keep_has_positive_data()