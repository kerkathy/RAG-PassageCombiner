# %%
## built-in
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
from tqdm import tqdm

## own
from utils import (
    get_yaml_file,
    set_seed,
    make_index,
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

    config_file = 'config/build_index_nq.yaml'
    yaml_config = get_yaml_file(config_file)
    # yaml_config = get_yaml_file(args.config_file)
    # args_dict = {k:v for k,v in vars(args).items() if v is not None}
    args_dict = {}
    args_dict['config_file'] = config_file
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

# %%
def main():
    # %%
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        # device_placement='cpu' if debug else 'auto',  # Change this line
        log_with=None,  # Change this line
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )
    # print the device
    logger.info(f"device: {accelerator.device}")

    logger.info(f"...Loading retriever model {args.retriever_model}...")
    if "dpr" in args.retriever_model:
        ret_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.retriever_model)
        doc_encoder = DPRContextEncoder.from_pretrained(args.retriever_model)
    else:
        ret_tokenizer = BertTokenizer.from_pretrained(args.retriever_model)
        doc_encoder = BertModel.from_pretrained(args.retriever_model,add_pooling_layer=False)

    logger.info("...Loading train data...")
    train_data = json.load(open(args.train_file))
    logger.info(f"Size of train data: {len(train_data)}")

    logger.info("...Creating train Corpus...")
    train_corpus = [[x['text'] for x in sample['ctxs']] for sample in train_data]
    logger.info(f"Size of train corpus: {len(train_corpus)}")

    # prepare doc_encoder
    logger.info(f"...Preparing doc encoder...")
    doc_encoder = accelerator.prepare(doc_encoder)
    logger.info(f"doc_encoder is on {doc_encoder.device}")

    if not os.path.exists(args.train_index_path):
        logger.info(f"...Creating index...")
        logger.info(f"...Creating train index with size {len(train_corpus)}...")
        train_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(train_corpus)]
        torch.save(train_doc_embeddings, args.train_index_path)
        logger.info(f"TRAIN Index saved to {args.train_index_path}")

    # logger.info("...Loading dev data...")
    # dev_data = json.load(open(args.dev_file))
    # logger.info("...Creating dev Corpus...")
    # dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]

    # if not os.path.exists(args.dev_index_path):
    #     logger.info(f"...Creating dev index with size {len(dev_corpus)}...")
    #     dev_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
    #     torch.save(dev_doc_embeddings, args.dev_index_path)
    #     logger.info(f"DEV Index saved to {args.dev_index_path}")

# %%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn') # try to fix the cuda init error when we put query encoder on cuda
    main()