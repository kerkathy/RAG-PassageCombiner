# %%
## built-in
import json,os
import types
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

class Index:
    def __init__(self, args):
        self.args = args
        self.train_doc_embeddings = None
        self.dev_doc_embeddings = None
        self.empty_doc_embedding = None

    def create(self, train=False, dev=False, empty=False):
        """
        Choose to create the index for train or dev data or both
        """
        if not train and not dev and not empty:
            return
        if "dpr" == self.args.encoder_type:
            ret_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.args.retriever_model, cache_dir=self.args.cache_dir)
            doc_encoder = DPRContextEncoder.from_pretrained(self.args.retriever_model, cache_dir=self.args.cache_dir)
        else:
            ret_tokenizer = BertTokenizer.from_pretrained(self.args.retriever_model, cache_dir=self.args.cache_dir)
            doc_encoder = BertModel.from_pretrained(self.args.retriever_model, add_pooling_layer=False, cache_dir=self.args.cache_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        doc_encoder.to(device)
        doc_encoder.eval()

        if train:
            print(f"...Loading data from {self.args.train_file}...")
            train_data = json.load(open(self.args.train_file))
            print(f"Size of train data: {len(train_data)}")
            print("...Creating Corpus...")
            train_corpus = [[x['text'] for x in sample['ctxs']] for sample in train_data]
            print(f"Size of train corpus: {len(train_corpus)}")
            self.train_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(train_corpus)]
            print(f"Train doc embedding calculated")
        if dev:
            print(f"...Loading data from {self.args.dev_file}...")
            dev_data = json.load(open(self.args.dev_file))
            print(f"Size of dev data: {len(dev_data)}")
            print("...Creating Corpus...")
            dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]
            print(f"Size of dev corpus: {len(dev_corpus)}")
            self.dev_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
            print(f"DEV Index calculated")
        if empty:
            self.empty_doc_embedding = make_index(["[UNK]"], ret_tokenizer, doc_encoder).squeeze()
            print(f"Empty doc embedding calculated")

    def read(self, train=False, dev=False, empty=False):
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
    
    def save(self):
        if self.train_doc_embeddings is not None:
            if os.path.exists(self.args.train_index_path):
                print(f"File {self.args.train_index_path} already exists. Not overwriting.")
            else:
                torch.save(self.train_doc_embeddings, self.args.train_index_path)
                print(f"Saved the train embeddings to {self.args.train_index_path}")

        if self.dev_doc_embeddings is not None:
            if os.path.exists(self.args.dev_index_path):
                print(f"File {self.args.dev_index_path} already exists. Not overwriting.")
            else:
                torch.save(self.dev_doc_embeddings, self.args.dev_index_path)
                print(f"Saved the dev embeddings to {self.args.dev_index_path}")

        if self.empty_doc_embedding is not None:
            if os.path.exists(self.args.empty_index_path):
                print(f"File {self.args.empty_index_path} already exists. Not overwriting.")
            else:
                torch.save(self.empty_doc_embedding, self.args.empty_index_path)
                print(f"Saved the empty embeddings to {self.args.empty_index_path}")

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
        self.read(train=True if "train" in to_read else False, dev=True if "dev" in to_read else False, empty=True if "empty" in to_read else False)
        if self.args.extract:
            self.extract()
        if self.args.normalize:
            self.normalize()
        self.save()

if __name__ == '__main__':
    config_file = 'config/preprocess_idx.yaml'
    yaml_config = get_yaml_file(config_file)
    args_dict = {}
    args_dict['config_file'] = config_file

    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    set_seed(args.seed)

    index = Index(args)
    index.process_and_save()