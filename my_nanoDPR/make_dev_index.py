# %%
## built-in
import time,random,queue,sys
import math,logging,json,random,os
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud
os.environ["CUDA_LAUNCH_BLOCKING"]="1" ## for debugging

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
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
import torch.distributed as dist
from tqdm import tqdm
import wandb

## own
from utils import (
    get_yaml_file,
    set_seed,
    get_linear_scheduler,
    normalize_query,
    make_index,
    retrieve_top_k_docid,
    load_lm_model_and_tokenizer,
    get_lm_score,
)

debug = False  # set log mode to debug, and stop wandb logging

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
logger = get_logger(__name__)

def parse_args():
    # import argparse
    # parser = argparse.ArgumentParser()
    # ## adding args here for more control from CLI is possible
    # parser.add_argument("--config_file",default='config/train_dpr_nq.yaml')
    # args = parser.parse_args()

    config_file = 'config/train_dpr_nq.yaml'
    yaml_config = get_yaml_file(config_file)
    # yaml_config = get_yaml_file(args.config_file)
    # args_dict = {k:v for k,v in vars(args).items() if v is not None}
    args_dict = {}
    args_dict['config_file'] = config_file
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

class DualEncoder(nn.Module):
    def __init__(self,query_encoder,doc_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        # TODO: whether to freeze the doc encoder

    def forward(
        self,
        query_inputs, # each [bs,seq_len]
        doc_inputs, # each [bs*n_comb,seq_len]
    ):  
        CLS_POS = 0
        ## [bs,n_dim]
        query_embedding = self.query_encoder(**query_inputs).last_hidden_state[:,CLS_POS,:]
        
        ## [bs*n_comb,n_dim]
        doc_embedding = self.doc_encoder(**doc_inputs).last_hidden_state[:,CLS_POS,:]
        
        return query_embedding,doc_embedding  # [bs,n_dim], [bs*n_comb,n_dim]

def calculate_dpr_loss(matching_score,labels):
    return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

def calculate_KL_div_loss(
    input_logits, # [n_question,n_comb]
    target_logits, # [n_question,n_comb]
    temperature,
):
    """
    Calculate KL divergence loss between input and target logits
    Note: input_logits and target_logits are logits, not distributions
    """
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    loss = kl_loss(
        F.log_softmax(input_logits / temperature, dim=1),
        F.softmax(target_logits / temperature, dim=1),
    )
    return loss

class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, stage, args, accelerator):
        self.data = data
        self.doc_embeddings = doc_embeddings
        self.ret_tokenizer = ret_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.query_encoder = query_encoder
        self.stage = stage
        self.args = args
        # self.device = device
        self.accelerator = accelerator
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        corpus = [x['text'] for x in sample['ctxs']]
        doc_embeddings = self.doc_embeddings[idx]  # Move to correct device
        # doc_embeddings = self.doc_embeddings[idx].to(self.device)  # Move to correct device
        data = [(normalize_query(sample['question']), "", sample['answers'][0])] # tmp fix, only use the first answer 
        cur_prompt_ans = queue.Queue()
        next_prompts_ans = queue.Queue()
        for d in data:
            next_prompts_ans.put((d[0], d[2]))
        for _ in range(self.args.num_round):
            cur_prompt_ans, next_prompts_ans = next_prompts_ans, cur_prompt_ans
            while not cur_prompt_ans.empty():
                prompt, answer = cur_prompt_ans.get()
                doc_ids = retrieve_top_k_docid(prompt, doc_embeddings, self.ret_tokenizer, self.query_encoder, self.args.k, self.accelerator)  # Pass device to function
                for docid in doc_ids:
                    doc = corpus[docid]
                    data.append((prompt, doc, answer))
                    next_prompts_ans.put((" ".join([prompt, doc]), answer))

        return data # List of tuples
    def collate_fn(self, samples):
        """
        samples: List[List[tuple]]
        """
        # flatten the samples into a list of tuples
        logger.debug(f"Original batch size: {len(samples)}")
        samples = [item for sublist in samples for item in sublist]
        logger.debug(f"Real batch size: {len(samples)}")
        
        # Tokenize the data
        query_inputs = self.ret_tokenizer([x[0] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
        doc_inputs = self.ret_tokenizer([x[1] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
        prompt_ans_lm_inputs = self.lm_tokenizer(
            [" ".join([x[0], x[1]]) for x in samples], 
            [x[2] for x in samples], 
            max_length=256, padding=True, truncation=True, return_tensors='pt',
            return_token_type_ids=True
        )

        return {
            "query_inputs": query_inputs,
            "doc_inputs": doc_inputs,
            "prompt_ans_lm_inputs": prompt_ans_lm_inputs,
        }


# %%
def main():
    # %%
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        # device_placement='cpu' if debug else 'auto',  # Change this line
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,  # Change this line
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )
    # print the device
    logger.info(f"device: {accelerator.device}")

    logger.info("...Loading retriever models...")
    if "dpr" in args.retriever_model:
        ret_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.retriever_model)
        doc_encoder = DPRContextEncoder.from_pretrained(args.retriever_model)
    else:
        ret_tokenizer = BertTokenizer.from_pretrained(args.retriever_model)
        doc_encoder = BertModel.from_pretrained(args.retriever_model,add_pooling_layer=False)

    logger.info("...Loading data...")
    dev_data = json.load(open(args.dev_file))

    logger.info(f"Size of train data: {len(dev_data)}")

    logger.info("...Creating Corpus...")
    dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]
    logger.info(f"Size of train corpus: {len(dev_corpus)}")

    # prepare doc_encoder
    logger.info(f"...Preparing doc encoder...")
    doc_encoder = accelerator.prepare(doc_encoder)
    logger.info(f"doc_encoder is on {doc_encoder.device}")

    if not os.path.exists(args.dev_index_path):
        logger.info(f"...Creating index...")
        logger.info(f"...Creating train index with size {len(dev_corpus)}...")
        dev_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
        torch.save(dev_doc_embeddings, args.dev_index_path)
        logger.info(f"TRAIN Index saved to {args.dev_index_path}")

    accelerator.end_training()

# %%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn') # try to fix the cuda init error when we put query encoder on cuda
    main()