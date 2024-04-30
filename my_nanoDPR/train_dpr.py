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
        query_input_ids, # [bs,seq_len]
        query_attention_mask, # [bs,seq_len]
        query_token_type_ids, # [bs,seq_len],
        doc_input_ids, # [bs*n_comb,seq_len]
        doc_attention_mask, # [bs*n_comb,seq_len]
        doc_token_type_ids, # [bs*n_comb,seq_len]
    ):  
        CLS_POS = 0
        ## [bs,n_dim]
        query_embedding = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask = query_attention_mask,
            token_type_ids = query_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        ## [bs*n_comb,n_dim]
        doc_embedding = self.doc_encoder(
            input_ids = doc_input_ids,
            attention_mask = doc_attention_mask,
            token_type_ids = doc_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        return query_embedding,doc_embedding  # [bs,n_dim], [bs*n_comb,n_dim]

def calculate_dpr_loss(matching_score,labels):
    return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

def calculate_KL_div_loss(
    input_dstr, # [bs,comb]
    target_dstr, # [bs,comb]
    temperature,
):
    # Calculate KL divergence loss
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    loss = kl_loss(
        F.log_softmax(input_dstr / temperature, dim=1),
        F.softmax(target_dstr / temperature, dim=1),
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
        self.device = device  # Add this line
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        corpus = [x['text'] for x in sample['ctxs']]
        doc_embeddings = self.doc_embeddings[idx]
        data = [(normalize_query(sample['question']), "", sample['answers'][0])] # tmp fix, only use the first answer 
        # data = [(normalize_query(sample['question']), "", ans) for ans in sample['answers']] # turn each answer into a separate data point
        # print("data: ", data)
        cur_prompt_ans = queue.Queue()
        next_prompts_ans = queue.Queue()
        for d in data:
            next_prompts_ans.put((d[0], d[2]))
        for _ in range(self.args.num_round):
            cur_prompt_ans, next_prompts_ans = next_prompts_ans, cur_prompt_ans
            while not cur_prompt_ans.empty():
                prompt, answer = cur_prompt_ans.get()
                doc_ids = retrieve_top_k_docid(prompt, doc_embeddings, self.ret_tokenizer, self.query_encoder, self.args.k, self.device)
                for docid in doc_ids:
                    doc = corpus[docid]
                    # print("docid: ", docid)
                    # print("doc: ", doc)
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
        
        features = ['input_ids', 'attention_mask', 'token_type_ids']

        # Tokenize the data
        query_ret_results = self.ret_tokenizer([x[0] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
        doc_ret_results = self.ret_tokenizer([x[1] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
        prompt_ans_lm_results = self.lm_tokenizer(
            [" ".join([x[0], x[1]]) for x in samples], 
            [x[2] for x in samples], 
            max_length=256, padding=True, truncation=True, return_tensors='pt'
        )

        mapping = {"query_ret_": query_ret_results, "doc_ret_": doc_ret_results, "prompt_ans_lm_": prompt_ans_lm_results}
        collate_dict ={}

        for k, v in mapping.items():
            for feature in features:
                if feature in v:
                    collate_dict[k+feature] = v[feature]

        return collate_dict
            
            
def validate(model,dataloader,accelerator):
    model.eval()
    query_embeddings = []
    positive_doc_embeddings = []
    negative_doc_embeddings = []
    for batch in dataloader:
        with torch.no_grad():
            query_embedding,doc_embedding  = model(**batch)
        query_num,_ = query_embedding.shape
        query_embeddings.append(query_embedding.cpu())
        positive_doc_embeddings.append(doc_embedding[:query_num,:].cpu())
        negative_doc_embeddings.append(doc_embedding[query_num:,:].cpu())

    query_embeddings = torch.cat(query_embeddings,dim=0)
    doc_embeddings = torch.cat(positive_doc_embeddings+negative_doc_embeddings,dim=0)
    matching_score = torch.matmul(query_embeddings,doc_embeddings.permute(1,0)) # bs, num_pos+num_neg
    labels = torch.arange(query_embeddings.shape[0],dtype=torch.int64).to(matching_score.device)
    loss = calculate_dpr_loss(matching_score,labels=labels).item()
    ranks = calculate_average_rank(matching_score,labels=labels)
    
    if accelerator.use_distributed and accelerator.num_processes>1:
        ranks_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(ranks_from_all_gpus,ranks)
        ranks = [x for y in ranks_from_all_gpus for x in y]

        loss_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(loss_from_all_gpus,loss)
        loss = sum(loss_from_all_gpus)/len(loss_from_all_gpus)
    
    return sum(ranks)/len(ranks),loss

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
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=4,pin_memory=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn,num_workers=4,pin_memory=True)
    
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
            # start of debug
            for k, v in batch.items():
                if "ret_input_ids" in k:
                    print(f"{k}: ", ret_tokenizer.batch_decode(v, skip_special_tokens=True), "\n")
                if "lm_input_ids" in k:
                    print(f"{k}: ", lm_tokenizer.batch_decode(v, skip_special_tokens=True), "\n")
            break
            # end of debug


            # logger.debug(f"... Loading step {step} ...")
            # prompt_ans_lm_input_ids = batch['prompt_ans_lm_input_ids'].to(accelerator.device)
            # prompt_ans_lm_attention_mask = batch['prompt_ans_lm_attention_mask'].to(accelerator.device)
            # prompt_ans_lm_token_type_ids = batch['prompt_ans_lm_token_type_ids'].to(accelerator.device)

            # with accelerator.accumulate(dual_encoder):
            #     with accelerator.autocast():
            #         # remove prompt_lm_input_ids, answer_lm_input_ids from batch
            #         for key in ['prompt_ans_lm_input_ids','prompt_ans_lm_attention_mask','prompt_ans_lm_token_type_ids']:
            #             del batch[key]
                    
            #         logger.debug("Sending batch to model...")
            #         query_embedding,doc_embedding  = dual_encoder(**batch)
            #         single_device_query_num,_ = query_embedding.shape
            #         single_device_doc_num,_ = doc_embedding.shape

            #         logger.debug("Waiting for everyone...")
            #         if accelerator.use_distributed:
            #             doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
            #             dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
            #             doc_list[dist.get_rank()] = doc_embedding
            #             doc_embedding = torch.cat(doc_list, dim=0)

            #             query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
            #             dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
            #             query_list[dist.get_rank()] = query_embedding
            #             query_embedding = torch.cat(query_list, dim=0)

            #         matching_score = torch.matmul(query_embedding,doc_embedding.permute(1,0)) # [bs,bs*n_comb]
            #         labels = torch.cat([torch.arange(single_device_query_num) + gpu_index * single_device_doc_num for gpu_index in range(accelerator.num_processes)],dim=0).to(matching_score.device) # [bs]
                    
            #         # Select only necessary columns from labels
            #         bs = single_device_query_num
            #         n_comb = single_device_doc_num // bs
            #         indices = torch.arange(bs*n_comb).view(bs, n_comb)
            #         retrieval_dstr = retrieval_dstr[torch.arange(bs).unsqueeze(1), indices] # [bs,n_comb]

            #         logger.debug("Calculating loss from LM...")
            #         lm_score = get_lm_score(
            #             language_model, 
            #             input_ids=prompt_ans_lm_input_ids,
            #             attention_mask=prompt_ans_lm_attention_mask,
            #             token_type_ids=prompt_ans_lm_token_type_ids,
            #             max_length=model_max_length,
            #             max_tokens_to_generate=args.max_tokens,
            #             num_orig_question=args.per_device_train_batch_size,
            #             temperature=args.temperature,
            #         )
            #         loss = calculate_KL_div_loss(retrieval_dstr=retrieval_dstr, lm_dstr=lm_score, temperature=args.temperature)

            #     accelerator.backward(loss)

            #     ## one optimization step
            #     if accelerator.sync_gradients:
            #         progress_bar.update(1)
            #         progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
            #         completed_steps += 1
            #         accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
            #         if not accelerator.optimizer_step_was_skipped:
            #             lr_scheduler.step()
            #         accelerator.log({"training_loss": loss}, step=completed_steps)
            #         accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
            #         if completed_steps % EVAL_STEPS == 0:
            #             avg_rank,loss = validate(dual_encoder,dev_dataloader,accelerator)
            #             dual_encoder.train()
            #             accelerator.log({"avg_rank": avg_rank, "loss":loss}, step=completed_steps)
            #             accelerator.wait_for_everyone()
            #             if accelerator.is_local_main_process:
            #                 unwrapped_model = accelerator.unwrap_model(dual_encoder)
            #                 unwrapped_model.query_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
            #                 ret_tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/query_encoder"))
                            
            #                 unwrapped_model.doc_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/doc_encoder"))
            #                 ret_tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}/doc_encoder"))

            #             accelerator.wait_for_everyone()
                
            #     optimizer.step()
            #     optimizer.zero_grad()
    
    if not debug and accelerator.is_local_main_process:
        wandb_tracker.finish()
    
    accelerator.end_training()
# %%
if __name__ == '__main__':
    main()