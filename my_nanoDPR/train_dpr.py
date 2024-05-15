# %%
## built-in
import time,random,queue,sys
import math,logging,json,random,os,psutil
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud
os.environ["CUDA_LAUNCH_BLOCKING"]="1" ## for debugging

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

## own
from utils import (
    ensure_directory_exists_for_file,
    get_yaml_file,
    set_seed,
    get_linear_scheduler,
    normalize_query,
    make_index,
    load_lm_model_and_tokenizer,
    load_doc_encoder_and_tokenizer,
    load_query_encoder_and_tokenizer,
)
from train_utils import (
    train_and_eval,
    validate,
)

debug = False  # set log mode to debug, and stop wandb logging
max_ret_token_len = 0
max_lm_token_len = 0

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
logger = get_logger(__name__)

def parse_args():
    # config_file = 'config/train_dpr_nq.yaml'
    # config_file = 'config/llama_train_dpr_nq.yaml'
    # yaml_config = get_yaml_file(config_file)
    # args_dict = {}
    # args_dict['config_file'] = config_file

    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/train_dpr_nq.yaml')
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config) # access in attribute style
    return args

class QADataset(torch.utils.data.Dataset):
    def __init__(self, qa_pairs, all_corpus, all_doc_embeddings):
        self.qa_pairs = qa_pairs
        self.all_corpus = all_corpus
        self.all_doc_embeddings = all_doc_embeddings
        
    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        data = [self.qa_pairs[idx]]  # each item is (query, all_doc, answer, last_doc_embedding)
        corpus = self.all_corpus[idx]
        doc_embeddings = self.all_doc_embeddings[idx]  # Move to correct device
        return {"data": data, "corpus": corpus, "doc_embeddings": doc_embeddings}
    
    def collate_fn(self, samples):
        """
        samples: List[Dict]
        """
        return samples

# %%
def main():
    # %%
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        # device_placement='cpu' if debug else 'auto',  # Change this line
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None if debug else 'wandb',  # Change this line
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )
    logger.debug("*** IN DEBUG MODE ***")
    logger.info(f"device: {accelerator.device}")
    model_short_name = "flan" if "flan" in args.lm_model else "llama"
    
    if args.resume_training:
        assert os.path.exists(args.resume_path), f"resume_path {args.resume_path} does not exist"
        logger.info(f"Resuming training from {args.resume_path}")
        # init tracker without config
        accelerator.init_trackers(
            project_name="dpr",
            # config=args,
            init_kwargs={"wandb":{"id":args.resume_wandb_id, "resume":"must"}},
            # init_kwargs={"wandb":{"allow_val_change": True, "id":args.resume_wandb_id, "resume":"must"}},
        )
    else:
        accelerator.init_trackers(
            project_name="dpr", 
            config=args,
            init_kwargs={"wandb":{"name":
                f"({args.data_size}) {model_short_name}-{args.max_round}round-{args.k}k-bs({args.per_device_train_batch_size}&{args.per_device_eval_batch_size})({args.train_llm_batch_size}&{args.eval_llm_batch_size})"}}
        )
    # %%
    if not debug and accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir
        wandb_tracker.run.log_code(".")
        if not args.resume_training:
            wandb_tracker.run.tags = [
                f"size: {args.data_size}", f"lm: {args.lm_model}", 
                f"query_enc: {args.query_encoder}", f"doc_enc: {args.doc_encoder}", 
                f"max_round: {args.max_round}", f"k: {args.k}", 
                f"train_bs: {args.per_device_train_batch_size}", f"eval_bs: {args.per_device_eval_batch_size}",
                f"temp: {args.temperature}","newline_format_prompt", "train", 
            ]
        else:
            # make sure current param is the same as the resumed one
            # except for resume_training, resume_path, resume_wandb_id
            exception_keys = ["resume_training", "resume_path", "resume_wandb_id"]
            for k,v in vars(args).items():
                if k not in exception_keys:
                    assert wandb_tracker.run.config[k] == v, \
                    f"config {k} is different from resumed one: {wandb_tracker.run.config[k]} != {v}"
            assert args.resume_wandb_id in args.resume_path, f"resume_wandb_id not in resume_path: {args.resume_wandb_id} not in {args.resume_path}"
    else:
        # TODO 改回來
        LOG_DIR = "./tmp_log_check_ckpt"  # Or any other directory you want to use when debugging
        # LOG_DIR = "./tmp_log"  # Or any other directory you want to use when debugging
    # %%
    query_tokenizer, query_encoder = load_query_encoder_and_tokenizer(args, logger)

    # %%
    if not debug and accelerator.is_local_main_process:
        wandb_tracker.run.watch(query_encoder, log_freq=500)
    # %%

    logger.info("...Loading language models...")
    language_model, lm_tokenizer, lm_config = load_lm_model_and_tokenizer(
        args.lm_model, device=accelerator.device, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    language_model.eval()

    model_max_length = lm_config.n_positions if hasattr(lm_config, "n_positions") else lm_config.max_position_embeddings
    # only pad if model is gpt2
    if "gpt2" in args.lm_model or "llama" in args.lm_model:
        lm_tokenizer.pad_token = "[PAD]"
        lm_tokenizer.padding_side = "left"
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

    if args.data_size == "debug":
        train_size, dev_size = 50, 10
    elif args.data_size == "1/10":
        train_size, dev_size = 10000, 1000
    elif args.data_size == "full":
        train_size, dev_size = 79168, 8757
    else:
        raise ValueError(f"Invalid data_size: {args.data_size}")
    args.train_file = args.train_file.replace(".json", f".size-{train_size}.json")
    args.dev_file = args.dev_file.replace(".json", f".size-{dev_size}.json")

    logger.info("...Loading data...")
    # skip data used as exemplars
    train_data = json.load(open(os.path.join(args.train_dir, args.train_file)))[args.num_exemplars:]
    dev_data = json.load(open(os.path.join(args.dev_dir, args.dev_file)))
    logger.info(f"Size of train data: {len(train_data)}")
    logger.info(f"Size of dev data: {len(dev_data)}")

    logger.info("...Creating Corpus...")
    train_corpus = [[x['text'] for x in sample['ctxs']] for sample in train_data]
    dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]
    logger.info(f"Size of train corpus: {len(train_corpus)}")
    logger.info(f"Size of dev corpus: {len(dev_corpus)}")

    train_index_path = os.path.join(args.index_dir, f"train_{train_size}.pt")
    dev_index_path = os.path.join(args.index_dir, f"dev_{dev_size}.pt")
    empty_doc_embedding_path = os.path.join(args.index_dir, "empty_doc.pt")

    if os.path.exists(train_index_path) and os.path.exists(dev_index_path) and os.path.exists(empty_doc_embedding_path):
        logger.info(f"...Loading index from {train_index_path} and {dev_index_path}...") 
        # skip those exemplars
        train_doc_embeddings = torch.load(train_index_path)[args.num_exemplars:]
        dev_doc_embeddings = torch.load(dev_index_path)
        empty_doc_embedding = torch.load(empty_doc_embedding_path)
        assert len(train_doc_embeddings) == len(train_corpus), f"len(train_doc_embeddings) ({len(train_doc_embeddings)}) != len(train_corpus), ({len(train_corpus)})"
        assert len(dev_doc_embeddings) == len(dev_corpus), f"len(dev_doc_embeddings) ({len(dev_doc_embeddings)}) != len(dev_corpus), ({len(dev_corpus)})"
    else:
        doc_tokenizer, doc_encoder = load_doc_encoder_and_tokenizer(args, logger)
        doc_encoder = accelerator.prepare(doc_encoder)
        logger.info(f"doc_encoder is on {doc_encoder.device}")
        logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

        with torch.no_grad():
            if not os.path.exists(args.train_index_path):
                logger.info(f"...Creating train index with size {len(train_corpus)}...")
                train_doc_embeddings = [make_index(corpus, doc_tokenizer, doc_encoder) for corpus in tqdm(train_corpus)]
                torch.save(train_doc_embeddings, args.train_index_path)
            if not os.path.exists(args.dev_index_path):
                logger.info(f"...Creating dev index with size {len(dev_corpus)}...")
                dev_doc_embeddings = [make_index(corpus, doc_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
                torch.save(dev_doc_embeddings, args.dev_index_path)
            if not os.path.exists(args.empty_doc_embedding_path):
                logger.info(f"...Creating empty embedding ...")
                empty_doc_embedding = make_index(["[UNK]"], doc_tokenizer, doc_encoder).squeeze() # for empty document
                torch.save(empty_doc_embedding, args.empty_doc_embedding_path)
        logger.info(f"Index saved to {args.train_index_path}, {args.dev_index_path}, {args.empty_doc_embedding_path}")
        logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

        logger.info("...Deleting doc_encoder...")
        doc_encoder = doc_encoder.to("cpu")
        del doc_encoder
        torch.cuda.empty_cache()
        logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

    gold_path = os.path.join(LOG_DIR, args.gold_dev_answers_path)
    if not os.path.exists(gold_path):
        logger.info(f"...Creating gold answers for dev set...")
        ensure_directory_exists_for_file(gold_path)
        gold_answers = []
        for sample in dev_data:
            gold_answers.append(sample['answers']) # log all answer 
            # gold_answers.append(sample['answers'][0])
        with open(gold_path, "w") as f:
            for ans in gold_answers:
                f.write(str(ans) + "\n")
        logger.info(f"Gold answers saved to {gold_path}")
        del gold_answers

    # TODO add feature of empty doc representation
    train_qa_pairs = [(normalize_query(sample['question']), [""], sample['answers'][0], empty_doc_embedding) for sample in train_data]
    dev_qa_pairs = [(normalize_query(sample['question']), [""], sample['answers'][0], empty_doc_embedding) for sample in dev_data]

    logger.info("...Build Dataset & Dataloader...")
    query_encoder = accelerator.prepare(query_encoder)
    logger.info(f"query_encoder is on {query_encoder.device}")
    train_dataset = QADataset(train_qa_pairs, train_corpus, train_doc_embeddings)
    dev_dataset = QADataset(dev_qa_pairs, dev_corpus, dev_doc_embeddings)
    
    logger.info("...Deleting train_data and dev_data...")
    del train_data, dev_data

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=args.num_workers,pin_memory=args.pin_memory)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn,num_workers=args.num_workers,pin_memory=args.pin_memory)
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in query_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in query_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_eps)
    
    logger.info("...Prepare accelerator...")
    optimizer, train_dataloader, dev_dataloader, language_model = accelerator.prepare(
        optimizer, train_dataloader, dev_dataloader, language_model 
    )
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    if isinstance(args.warmup_steps, float):
        args.warmup_steps = int(args.warmup_steps * MAX_TRAIN_STEPS)
        logger.info(f"Converted warmup_steps to {args.warmup_steps}")
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)
    completed_steps = 0

    # TODO: debug
    if args.resume_training:
        logger.info(f"...Loading old state_dict from ckpt {args.resume_path}...")
        state_dict = torch.load(args.resume_path)
        query_encoder.load_state_dict(state_dict["query_encoder"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        completed_steps = state_dict["completed_steps"]
        # skip first batches if resuming
        steps_to_skip_in_epoch = completed_steps % NUM_UPDATES_PER_EPOCH
        skipped_data_loader = accelerator.skip_first_batches(train_dataloader, steps_to_skip_in_epoch)
        EPOCHS_TO_SKIP = completed_steps // NUM_UPDATES_PER_EPOCH
        logger.info(f"...State_dict at step {completed_steps} loaded to query_encoder, optimizer, lr_scheduler...")
    else:
        steps_to_skip_in_epoch = 0
        EPOCHS_TO_SKIP = 0
        logger.info(f"\n...0 Step Evaluation...")
        steps_log_dir = os.path.join(LOG_DIR,f"step-{completed_steps}")
        if not os.path.exists(steps_log_dir):
            os.makedirs(steps_log_dir)
        loss = validate(query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, steps_log_dir)
        accelerator.log({"eval":loss}, step=completed_steps)

    logger.info("\n***** Running training *****")
    logger.info(f"  Num workers = {args.num_workers}")
    logger.info(f"  pin_memory = {args.pin_memory}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}. First {EPOCHS_TO_SKIP} epochs are skipped.")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Extended train batch size (retriever batch size) = {args.per_device_train_batch_size * sum([args.k ** i for i in range(args.max_round + 1)])}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}. First {completed_steps} steps are skipped.")
    logger.info(f"  Num steps per evaluation = {EVAL_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    logger.info(f"  Train LM batch size = {args.train_llm_batch_size}")
    logger.info(f"  Eval LM batch size = {args.eval_llm_batch_size}")
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), initial=completed_steps, disable=not accelerator.is_local_main_process,ncols=100)

    start_time = time.time()

    if args.resume_training:
        for step,raw_batch in enumerate(skipped_data_loader):
            train_and_eval(query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
                accelerator, model_max_length, optimizer, lr_scheduler, logger,
                step, epoch, completed_steps, MAX_TRAIN_STEPS, EVAL_STEPS, LOG_DIR, progress_bar)

    for epoch in range(EPOCHS_TO_SKIP,MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        logger.info(f"[Before load train data] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        for step,raw_batch in enumerate(train_dataloader):
            train_and_eval(query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
                accelerator, model_max_length, optimizer, lr_scheduler, logger,
                step, epoch, completed_steps, MAX_TRAIN_STEPS, EVAL_STEPS, LOG_DIR, progress_bar)

    if accelerator.is_local_main_process:
        logger.info(f"Time spent: {time.time() - start_time} seconds")
        logger.info(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
        logger.info("...!!Congrats!! Training finished :) ...")
        logger.info(f"Checkpoint saved to {LOG_DIR}")
        if not debug:
            wandb_tracker.finish()
    
    accelerator.end_training()

# %%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn') # try to fix the cuda init error when we put query encoder on cuda
    main()