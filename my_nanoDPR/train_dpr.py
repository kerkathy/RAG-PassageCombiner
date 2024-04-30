# %%
## built-in
import math,logging,json,random,functools,os
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import queue

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
    get_yaml_file,
    set_seed,
    get_linear_scheduler,
    normalize_query,
    make_index,
    retrieve_top_k_docid,
    load_lm_model_and_tokenizer,
    get_lm_score,
)

debug = True  # Set this to False when you're done debugging

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

def calculate_hit_cnt(matching_score,labels):
    _, max_ids = torch.max(matching_score,1)
    return (max_ids == labels).sum()

def calculate_average_rank(matching_score,labels):
    _,indices = torch.sort(matching_score,dim=1,descending=True)
    ranks = []
    for idx,label in enumerate(labels):
        rank = ((indices[idx] == label).nonzero()).item() + 1  ##  rank starts from 1
        ranks.append(rank)
    return ranks

class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, stage, args, device):
        self.data = data
        self.doc_embeddings = doc_embeddings
        self.ret_tokenizer = ret_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.query_encoder = query_encoder
        self.stage = stage
        self.args = args
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        corpus = [x['text'] for x in sample['ctxs']]
        doc_embeddings = self.doc_embeddings[idx].to(self.device)  # Move to correct device
        data = [(normalize_query(sample['question']), "", sample['answers'][0])] # tmp fix, only use the first answer 
        cur_prompt_ans = queue.Queue()
        next_prompts_ans = queue.Queue()
        for d in data:
            next_prompts_ans.put((d[0], d[2]))
        for _ in range(self.args.num_round):
            cur_prompt_ans, next_prompts_ans = next_prompts_ans, cur_prompt_ans
            while not cur_prompt_ans.empty():
                prompt, answer = cur_prompt_ans.get()
                doc_ids = retrieve_top_k_docid(prompt, doc_embeddings, self.ret_tokenizer, self.query_encoder, self.args.k, self.device)  # Pass device to function
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
        samples = [item for sublist in samples for item in sublist]
        print("len(samples): ", len(samples))
        
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
            
def validate(args, dual_encoder, language_model, validation_dataloader, accelerator, model_max_length):
    dual_encoder.eval()
    language_model.eval()
    total_loss = 0
    num_batches = 0

    for step, batch in enumerate(validation_dataloader):
        with torch.no_grad():
            query_embedding, doc_embedding = dual_encoder(query_inputs=batch['query_inputs'], doc_inputs=batch['doc_inputs'])
            single_device_query_num, _ = query_embedding.shape
            single_device_doc_num, _ = doc_embedding.shape

            if accelerator.use_distributed:
                doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                doc_list[dist.get_rank()] = doc_embedding
                doc_embedding = torch.cat(doc_list, dim=0)

                query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                query_list[dist.get_rank()] = query_embedding
                query_embedding = torch.cat(query_list, dim=0)

            retriever_score = torch.sum(query_embedding * doc_embedding, dim=1)  # [bs]
            num_orig_question = args.per_device_eval_batch_size
            retriever_score = retriever_score.reshape(num_orig_question, -1)

            lm_score = get_lm_score(
                language_model, 
                **batch['prompt_ans_lm_inputs'],
                max_length=model_max_length,
                max_tokens_to_generate=args.max_tokens,
                num_orig_question=num_orig_question,
            )
            loss = calculate_KL_div_loss(input_logits=retriever_score, target_logits=lm_score, temperature=args.temperature)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

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

    accelerator.init_trackers(
        project_name="dpr", 
        config=args,
    )
        
    if not debug and accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir
    else:
        LOG_DIR = "/tmp"  # Or any other directory you want to use when debugging


    ret_tokenizer = BertTokenizer.from_pretrained(args.retriever_model)
    query_encoder = BertModel.from_pretrained(args.retriever_model,add_pooling_layer=False)
    doc_encoder = BertModel.from_pretrained(args.retriever_model,add_pooling_layer=False)
    dual_encoder = DualEncoder(query_encoder,doc_encoder)
    dual_encoder.train()

    language_model, lm_tokenizer, lm_config, lm_device = load_lm_model_and_tokenizer(
        args.lm_model, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    model_max_length = lm_config.n_positions if hasattr(lm_config, "n_positions") else lm_config.max_position_embeddings
    lm_tokenizer.pad_token = "[PAD]"
    lm_tokenizer.padding_side = "left"

    train_data = json.load(open(args.train_file))[:2] # for debugging
    dev_data = json.load(open(args.dev_file))[:2] # for debugging

    train_corpus = [[x['text'] for x in sample['ctxs']] for sample in train_data]
    dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]

    train_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in train_corpus]
    dev_doc_embeddings = [make_index(corpus, ret_tokenizer, doc_encoder) for corpus in dev_corpus]

    print("train_doc_embeddings[0].shape: ", train_doc_embeddings[0].shape)
    print("dev_doc_embeddings[0].shape: ", dev_doc_embeddings[0].shape, "\n")

    train_dataset = QADataset(train_data, train_doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, 'train', args, accelerator.device)
    dev_dataset = QADataset(dev_data, dev_doc_embeddings, ret_tokenizer, lm_tokenizer, query_encoder, 'dev', args, accelerator.device)
    
    # debug: set num_worker=0, pin_memory=False
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=0,pin_memory=False)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn,num_workers=0,pin_memory=False)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in dual_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in dual_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_eps)
    
    dual_encoder, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        dual_encoder, optimizer, train_dataloader, dev_dataloader
    )
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        logger.debug(f"... About to load batches in epoch {epoch} ...")
        for step,batch in enumerate(train_dataloader):
            logger.debug(f"... Successfully load batches in epoch {epoch} ...")
            with accelerator.accumulate(dual_encoder):
                with accelerator.autocast():
                    logger.debug("...Sending batch to model...")
                    logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
                    logger.debug(f"batch['doc_inputs']['input_ids']: {batch['doc_inputs']['input_ids'].shape}")
                    query_embedding,doc_embedding  = dual_encoder(query_inputs=batch['query_inputs'],doc_inputs=batch['doc_inputs'])
                    logger.debug(f"query_embedding.shape: {query_embedding.shape}")
                    logger.debug(f"doc_embedding.shape: {doc_embedding.shape}")
                    # shape of both query_embedding and doc_embedding: [bs,n_dim]
                    # here bs = n_comb * num_orig_question
                    single_device_query_num,_ = query_embedding.shape
                    single_device_doc_num,_ = doc_embedding.shape

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
                    num_orig_question = args.per_device_train_batch_size
                    retriever_score = retriever_score.reshape(num_orig_question, -1)
                    logger.debug(f"retriever_score.shape: {retriever_score.shape}")
                    logger.debug(f"retriever_score: {retriever_score}")

                    logger.debug("Calculating loss from LM...")
                    lm_score = get_lm_score(
                        language_model, 
                        **batch['prompt_ans_lm_inputs'],
                        max_length=model_max_length,
                        max_tokens_to_generate=args.max_tokens,
                        num_orig_question=num_orig_question,
                    )
                    logger.debug(f"lm_score: {lm_score}")
                    loss = calculate_KL_div_loss(input_logits=retriever_score, target_logits=lm_score, temperature=args.temperature)
                    logger.debug(f"loss: {loss}")

                accelerator.backward(loss)

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        loss = validate(args,dual_encoder,language_model,dev_dataloader,accelerator,model_max_length)
                        dual_encoder.train()
                        accelerator.log({"loss":loss}, step=completed_steps)
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(dual_encoder)
                            unwrapped_model.query_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            ret_tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            
                            unwrapped_model.doc_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/doc_encoder"))
                            ret_tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/doc_encoder"))

                        accelerator.wait_for_everyone()
                
                optimizer.step()
                optimizer.zero_grad()
    
    if not debug and accelerator.is_local_main_process:
        wandb_tracker.finish()
    
    accelerator.end_training()
    logger.debug("...!!Congrats!! Training finished :) ...")
# %%
if __name__ == '__main__':
    main()

# %%
# # debug
# import torch
# import torch.nn.functional as F
# # %%
# tmp = torch.tensor([[-123.1659, -120.9617,  -39.9989],
#     [ -54.0244,  -54.0071,  -50.6071]]
# )

# F.log_softmax(tmp, dim=1),


# # %%
