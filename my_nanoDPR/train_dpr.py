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
from transformers import (
    BertTokenizer,
    BertModel,
)
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
    retrieve_top_k_docid,
    load_lm_model_and_tokenizer,
    get_lm_score,
    evaluate_dataset,
)

debug = False  # set log mode to debug, and stop wandb logging
take_few_data = True  # train, dev = 200, 100. Set this to True to take only a few data for debugging

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
    def __init__(self, qa_pairs, all_corpus, all_doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, stage, args, accelerator):
        self.qa_pairs = qa_pairs
        self.all_corpus = all_corpus
        self.all_doc_embeddings = all_doc_embeddings
        self.ret_tokenizer = ret_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.query_encoder = query_encoder
        self.stage = stage
        self.args = args
        self.accelerator = accelerator
        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        data = [self.qa_pairs[idx]] # (query, doc, doc_embedding, answer)
        corpus = self.all_corpus[idx]
        doc_embeddings = self.all_doc_embeddings[idx]  # Move to correct device
        cur_prompt_ans = queue.Queue()
        next_prompts_ans = queue.Queue()
        embedding_device = data[0][3].device
        for d in data:
            next_prompts_ans.put((d[0], d[2]))
        for _ in range(self.args.max_round):
            cur_prompt_ans, next_prompts_ans = next_prompts_ans, cur_prompt_ans
            while not cur_prompt_ans.empty():
                prompt, answer = cur_prompt_ans.get()
                doc_ids = retrieve_top_k_docid(prompt, doc_embeddings, self.ret_tokenizer, self.query_encoder, self.args.k, self.accelerator)  # Pass device to function
                for docid in doc_ids:
                    next_prompts_ans.put((" ".join([prompt, corpus[docid]]), answer))
                    data.append((prompt, corpus[docid], answer, doc_embeddings[docid].to(embedding_device)))
        # logger.debug(f"After getitem, data size: {len(data)}")
        return data # List of tuples

    def collate_fn(self, samples):
        """
        samples: List[List[tuple]]
        """
        # flatten the samples into a list of tuples
        # logger.debug(f"Original batch size: {len(samples)}")
        samples = [item for sublist in samples for item in sublist]
        logger.debug(f"Real batch size: {len(samples)}")
        
        query_inputs = self.ret_tokenizer([x[0] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
        # collect doc_inputs from doc_embeddings
        doc_embeddings = torch.stack([x[3] for x in samples], dim=0)
        prompt_ans_lm_inputs = self.lm_tokenizer(
            [" ".join([x[0], x[1]]) for x in samples], 
            [x[2] for x in samples],
            max_length=256, padding=True, truncation=True, return_tensors='pt',
            return_token_type_ids=True
        )

        return {
            "query_inputs": query_inputs,
            "doc_embeddings": doc_embeddings,
            "prompt_ans_lm_inputs": prompt_ans_lm_inputs,
        }
    
def validate(
        query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
        accelerator, model_max_length, steps_log_dir
):
    logger.info("*** Start validation ***")
    query_encoder.eval()
    language_model.eval()
    total_loss = 0
    total_ans_prob = 0
    num_batches = 0
    all_retriever_pick = []

    for step, batch in enumerate(dev_dataloader):
        with torch.no_grad():
            ## Metric 1. Loss
            logger.debug("...Sending batch to model...")
            logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
            logger.debug(f"batch['doc_embeddings']: {batch['doc_embeddings'].shape}")
            logger.debug(f"query_encoder device: {next(query_encoder.parameters()).device}")
            query_embedding = query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:]
            doc_embedding = batch["doc_embeddings"]
            
            # logger.debug(f"query_embedding device: {query_embedding.device}; doc_embedding device: {doc_embedding.device}")
            logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            
            single_device_query_num, _ = query_embedding.shape
            single_device_doc_num = doc_embedding.shape[0]

            logger.debug("...Waiting for everyone...")
            if accelerator.use_distributed:
                doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                doc_list[dist.get_rank()] = doc_embedding
                doc_embedding = torch.cat(doc_list, dim=0)

                query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                query_list[dist.get_rank()] = query_embedding
                query_embedding = torch.cat(query_list, dim=0)

            logger.debug("...Calculating loss from DPR...")
            retriever_score = torch.sum(query_embedding * doc_embedding, dim=1)  # [bs]
            num_orig_question = single_device_query_num // sum([args.k ** i for i in range(args.max_round + 1)])
            retriever_score = retriever_score.reshape(num_orig_question, -1)
            logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            logger.debug("...Calculating loss from LM...")
            lm_score = get_lm_score(
                language_model, 
                accelerator.device,
                **batch['prompt_ans_lm_inputs'],
                max_length=model_max_length,
                max_tokens_to_generate=args.max_tokens,
                num_orig_question=num_orig_question,
                llm_batch_size=args.llm_batch_size,
            )
            logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            logger.debug(f"...Calculating loss...: {torch.cuda.memory_allocated() / 1e6} MB")
            loss = calculate_KL_div_loss(input_logits=retriever_score, target_logits=lm_score, temperature=args.temperature)
            total_loss += loss.item()
            logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            ## Metric 2. Average answer probability
            # for each question, take idx of max retriever_score 
            # get its corresponding lm_score
            # note. retriever_score and lm_score are both [n_question,n_comb]
            retrievers_pick = torch.argmax(retriever_score,dim=1) # [n_question]
            lm_score = lm_score[torch.arange(num_orig_question),retrievers_pick] # [n_question]
            total_ans_prob += lm_score.exp().mean().item()
            all_retriever_pick.extend(retrievers_pick.tolist())

            ## Metric 3. Exact match
            # reshape from [n_question*n_comb,seq_len] to [n_question,n_comb,seq_len]
            batch['prompt_ans_lm_inputs'] = {
                k: v.view(num_orig_question, -1, v.shape[-1]) for k,v in batch['prompt_ans_lm_inputs'].items()
            }
            # and then take the retriever's pick for each question
            batch['prompt_ans_lm_inputs'] = {
                k: v[torch.arange(num_orig_question),retrievers_pick] for k,v in batch['prompt_ans_lm_inputs'].items()
            }
            # calculate exact match
            d = evaluate_dataset(
                model=language_model, 
                tokenizer=lm_tokenizer,
                device=accelerator.device,
                max_length=model_max_length,
                prompt_ans_lm_inputs=batch['prompt_ans_lm_inputs'],
                max_tokens_to_generate=args.max_tokens,
                steps_log_dir=steps_log_dir,
                llm_batch_size=args.llm_batch_size,
            )

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_prob = total_ans_prob / num_batches
    logger.info(f"Validation loss: {avg_loss}")
    logger.info(f"Validation avg answer prob: {avg_prob}")
    return {"loss": avg_loss, "avg_prob": avg_prob, **d, "retriever_pick": all_retriever_pick}


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
    # print the device
    logger.info(f"device: {accelerator.device}")

    logger.debug("*** IN DEBUG MODE ***")

    accelerator.init_trackers(
        project_name="dpr", 
        config=args,
        init_kwargs={"wandb":{"name":"batch size=3"}},
    )
        
    if not debug and accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir
        wandb_tracker.run.log_code(".")
    else:
        LOG_DIR = "./tmp_log"  # Or any other directory you want to use when debugging

    logger.info("...Loading retriever models...")
    ret_tokenizer = BertTokenizer.from_pretrained(args.retriever_model)
    query_encoder = BertModel.from_pretrained(args.retriever_model,add_pooling_layer=False)
    doc_encoder = BertModel.from_pretrained(args.retriever_model,add_pooling_layer=False)
    doc_encoder.eval()
    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
    if not debug and accelerator.is_local_main_process:
        wandb_tracker.run.watch(query_encoder, log="all", log_freq=500)

    logger.info("...Loading language models...")
    language_model, lm_tokenizer, lm_config, lm_device = load_lm_model_and_tokenizer(
        args.lm_model, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    model_max_length = lm_config.n_positions if hasattr(lm_config, "n_positions") else lm_config.max_position_embeddings
    lm_tokenizer.pad_token = "[PAD]"
    lm_tokenizer.padding_side = "left"
    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

    logger.info("...Loading data...")
    train_data = json.load(open(args.train_file))

    if take_few_data:
        train_data = train_data[:50]

    if args.train_file == args.dev_file:
        # random split train and dev data
        random.shuffle(train_data)
        dev_data = train_data[:len(train_data)//10]
        train_data = train_data[len(train_data)//10:]
    else:
        dev_data = json.load(open(args.dev_file))
        if take_few_data:
            dev_data = dev_data[:10]

    logger.info(f"Size of train data: {len(train_data)}")
    logger.info(f"Size of dev data: {len(dev_data)}")

    logger.info("...Creating Corpus...")
    train_corpus = [[x['text'] for x in sample['ctxs']] for sample in train_data]
    dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]
    logger.info(f"Size of train corpus: {len(train_corpus)}")
    logger.info(f"Size of dev corpus: {len(dev_corpus)}")

    if os.path.exists(args.train_index_path) and os.path.exists(args.dev_index_path) and os.path.exists(args.empty_doc_embedding_path):
        logger.info(f"...Loading index from {args.train_index_path} and {args.dev_index_path}...") 
        train_doc_embeddings = torch.load(args.train_index_path)
        dev_doc_embeddings = torch.load(args.dev_index_path)
        empty_doc_embedding = torch.load(args.empty_doc_embedding_path)
        assert len(train_doc_embeddings) == len(train_corpus), f"len(train_doc_embeddings) ({len(train_doc_embeddings)}) != len(train_corpus), ({len(train_corpus)})"
        assert len(dev_doc_embeddings) == len(dev_corpus), f"len(dev_doc_embeddings) ({len(dev_doc_embeddings)}) != len(dev_corpus), ({len(dev_corpus)})"
    else:
        # prepare doc_encoder
        logger.info(f"...Preparing doc encoder...")
        doc_encoder = accelerator.prepare(doc_encoder)
        logger.info(f"doc_encoder is on {doc_encoder.device}")
        logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        logger.info(f"...Creating train index with size {len(train_corpus)}...")
        with torch.no_grad():
            train_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(train_corpus)]
            logger.info(f"...Creating dev index with size {len(dev_corpus)}...")
            dev_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
            # use [UNK]'s representation for empty document
            unk_inputs = ret_tokenizer("[UNK]", return_tensors='pt').to(doc_encoder.device)
            empty_doc_embedding = doc_encoder(**unk_inputs).last_hidden_state.mean(dim=1).squeeze(0)
        torch.save(train_doc_embeddings, args.train_index_path)
        torch.save(dev_doc_embeddings, args.dev_index_path)
        torch.save(empty_doc_embedding, args.empty_doc_embedding_path)
        logger.info(f"Index saved to {args.train_index_path}, {args.dev_index_path}, {args.empty_doc_embedding_path}")
        logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        logger.info("...Deleting doc_encoder...")
        del doc_encoder
        accelerator.empty_cache()
        logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

    gold_path = os.path.join(LOG_DIR, args.gold_dev_answers_path)
    if not os.path.exists(gold_path):
        logger.info(f"...Creating gold answers for dev set...")
        ensure_directory_exists_for_file(gold_path)
        gold_answers = []
        for sample in dev_data:
            gold_answers.append(sample['answers'][0])
        with open(gold_path, "w") as f:
            for ans in gold_answers:
                f.write(ans + "\n")
        logger.info(f"Gold answers saved to {gold_path}")
        del gold_answers

    train_qa_pairs = [(normalize_query(sample['question']), "", sample['answers'][0], empty_doc_embedding) for sample in train_data]
    dev_qa_pairs = [(normalize_query(sample['question']), "", sample['answers'][0], empty_doc_embedding) for sample in dev_data]

    logger.info("...Build Dataset & Dataloader...")
    query_encoder = accelerator.prepare(query_encoder)
    logger.info(f"query_encoder is on {query_encoder.device}")
    train_dataset = QADataset(train_qa_pairs, train_corpus, train_doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, 'train', args, accelerator)
    dev_dataset = QADataset(dev_qa_pairs, dev_corpus, dev_doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, 'dev', args, accelerator)
    
    logger.info("...Deleting train_data and dev_data...")
    logger.info(f"CPU memory used before deletion: {psutil.virtual_memory().used / 1e6} MB")
    del train_data, dev_data
    logger.info(f"CPU memory used after deletion: {psutil.virtual_memory().used / 1e6} MB")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=args.num_workers,pin_memory=args.pin_memory)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn,num_workers=args.num_workers,pin_memory=args.pin_memory)
    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
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
    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num workers = {args.num_workers}")
    logger.info(f"  pin_memory = {args.pin_memory}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Extended train batch size (retriever batch size) = {args.per_device_train_batch_size * sum([args.k ** i for i in range(args.max_round + 1)])}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Num steps per evaluation = {EVAL_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    logger.info(f"  LM batch size = {args.llm_batch_size}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    start_time = time.time()

    logger.info(f"...0 Step Evaluation...")
    steps_log_dir = os.path.join(LOG_DIR,f"step-{completed_steps}")
    if not os.path.exists(steps_log_dir):
        os.makedirs(steps_log_dir)
    loss = validate(query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, steps_log_dir)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        for step,batch in enumerate(train_dataloader):
            logger.debug(f"... Successfully load batches in epoch {epoch} ...")
            query_encoder.train()
            with accelerator.accumulate(query_encoder): # gradient accumulation
                with accelerator.autocast():
                    logger.debug("...Sending batch to model...")
                    logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
                    logger.debug(f"batch['doc_embeddings']: {batch['doc_embeddings'].shape}")
                    # logger.debug(f"query_encoder device: {next(query_encoder.parameters()).device}")
                    query_embedding = query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:]
                    doc_embedding = batch["doc_embeddings"]
                    
                    # logger.debug(f"query_embedding device: {query_embedding.device}; doc_embedding device: {doc_embedding.device}")
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                    # shape of both query_embedding and doc_embedding: [bs,n_dim]
                    # where bs = n_comb * num_orig_question
                    single_device_query_num,_ = query_embedding.shape
                    single_device_doc_num = doc_embedding.shape[0]

                    logger.debug("...Waiting for everyone...")
                    if accelerator.use_distributed:
                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)

                    logger.debug("...Calculating similarity score from DPR...")
                    retriever_score = torch.sum(query_embedding * doc_embedding, dim=1)  # [bs]
                    num_orig_question = single_device_query_num // sum([args.k ** i for i in range(args.max_round + 1)])
                    retriever_score = retriever_score.reshape(num_orig_question, -1)
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                    # move retriever_score to cpu
                    logger.debug(f"...Moving score to cpu and delete embeddings...")
                    retriever_score = retriever_score.cpu()

                    # delete embeddings
                    del query_embedding
                    del doc_embedding
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                    logger.debug(f"...Calculating loss from LM on {accelerator.device}...")
                    # very likely to OOM error here
                    lm_score = get_lm_score(
                        language_model,
                        accelerator.device,
                        **batch['prompt_ans_lm_inputs'],
                        max_length=model_max_length,
                        max_tokens_to_generate=args.max_tokens,
                        num_orig_question=num_orig_question,
                        llm_batch_size=args.llm_batch_size,
                    )
                    
                    # move retriever_score back to GPU
                    retriever_score = retriever_score.to(accelerator.device)
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                    logger.debug(f"retriever_score.device: {retriever_score.device}; lm_score.device: {lm_score.device}")
                    loss = calculate_KL_div_loss(input_logits=retriever_score, target_logits=lm_score, temperature=args.temperature)
                    del retriever_score, lm_score
                    logger.debug(f"Finish compute loss. loss = {loss}")
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                accelerator.backward(loss)

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0 or completed_steps == MAX_TRAIN_STEPS:
                        logger.info(f"...Evaluation...")
                        steps_log_dir = os.path.join(LOG_DIR,f"step-{completed_steps}")
                        if not os.path.exists(steps_log_dir):
                            os.makedirs(steps_log_dir)
                        loss = validate(query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, steps_log_dir)
                        query_encoder.train() # Make sure the model is back in training mode after validation
                        accelerator.log({"eval":loss}, step=completed_steps)
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            query_encoder_path = os.path.join(steps_log_dir, "query_encoder")
                            ensure_directory_exists_for_file(query_encoder_path)
                            query_encoder.save_pretrained(query_encoder_path)
                            ret_tokenizer.save_pretrained(query_encoder_path)
                            logger.info(f"Checkpoint saved to {LOG_DIR}")
                            
                        accelerator.wait_for_everyone()
                logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                
                optimizer.step()
                optimizer.zero_grad()
                logger.debug(f"Finish step {step}.\n GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
    logger.info(f"Time spent: {time.time() - start_time} seconds")
    logger.info(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
    logger.info("...!!Congrats!! Training finished :) ...")
    logger.info(f"Checkpoint saved to {LOG_DIR}")

    if not debug and accelerator.is_local_main_process:
        wandb_tracker.finish()
    
    accelerator.end_training()

# %%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn') # try to fix the cuda init error when we put query encoder on cuda
    main()