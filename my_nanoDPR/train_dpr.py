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
    retrieve_top_k_docid,
    load_lm_model_and_tokenizer,
    get_lm_score,
    get_t5_lm_score,
    lm_gen_and_check,
    load_doc_encoder_and_tokenizer,
    load_query_encoder_and_tokenizer,
    make_prompt,
    text_has_answer,
)

debug = True  # set log mode to debug, and stop wandb logging

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
        self.lm_name = lm_tokenizer.name_or_path
        self.query_encoder = query_encoder
        self.stage = stage
        self.args = args
        self.accelerator = accelerator

    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        data = [self.qa_pairs[idx]]  # each item is (query, all_doc, answer, last_doc_embedding)
        corpus = self.all_corpus[idx]
        doc_embeddings = self.all_doc_embeddings[idx]  # Move to correct device
        embedding_device = data[0][3].device
        # Initialize pointers
        cur_visited = 0
        this_round_should_visited = 0
        next_round_should_visited = len(data)

        # Loop over rounds
        for i_rnd in range(self.args.max_round):
            logger.debug(f"Round {i_rnd} has {next_round_should_visited} data to go thru...")
            # Update pointers
            this_round_should_visited = next_round_should_visited
            next_round_should_visited = 0
            cur_visited = 0
            # Process data from current round
            while cur_visited < this_round_should_visited:
                # Get current data
                query, doc_list, answer, _ = data[cur_visited]
                cur_visited += 1

                # Retrieve top k documents
                doc_ids = retrieve_top_k_docid(doc_list[-1] + " " + query, doc_embeddings, self.ret_tokenizer, self.query_encoder, self.args.k)
                # Append new data
                for docid in doc_ids:
                    new_doc_list = doc_list + [corpus[docid]]
                    data.append((query, new_doc_list, answer, doc_embeddings[docid].to(embedding_device)))

                    # Increment next_pointer
                    next_round_should_visited += 1
        logger.debug(f"After getitem, data size: {len(data)}")
        # for i in range(len(data)):
        #     logger.debug(f"Data {i}: {data[i][:-1]}")
        return data  # List of tuples


    def collate_fn(self, samples):
        """
        samples: List[List[tuple]]
        """
        # TODO add feature: 不同文章數量的分開 decode
        # flatten the samples into a list of tuples
        # logger.debug(f"Original batch size: {len(samples)}")
        samples = [item for sublist in samples for item in sublist]
        logger.debug(f"Real batch size: {len(samples)}")
        
        # each item is (query, all_doc, answer, last_doc_embedding)
        query_inputs = self.ret_tokenizer([x[0] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
        # collect doc_inputs from doc_embeddings
        doc_embeddings = torch.stack([x[3] for x in samples], dim=0)
        
        # in each item, num_docs shuold be num items where x[1][i] != "" (empty doc holder)
        prompt = [make_prompt(
            question=x[0], documents=x[1], lm_name=self.lm_name, 
            num_docs=len([doc for doc in x[1] if doc != ""]),
            num_exemplars=self.args.num_exemplars, dataset=self.args.dataset_name) for x in samples]
        answer = [x[2] for x in samples]
        num_has_answer = 0
        for a, p in zip(answer, prompt):
            logger.debug(f"Answer: {a}")
            logger.debug(f"Prompt: {p}")
            if text_has_answer(a, p):
                num_has_answer += 1

        # # debug: max prompt words and print that prompt
        # max_prompt_len = max([len(x.split()) for x in prompt])
        # max_prompt_idx = [i for i, x in enumerate(prompt) if len(x.split()) == max_prompt_len]
        # logger.debug(f"Max prompt len: {max_prompt_len}")
        # logger.debug(f"Max prompt: {prompt[max_prompt_idx[0]]}")

        if "t5" in self.lm_name:
            # separate input_ids (send into encoder) and labels (send into decoder)
            # regarding max_length: https://huggingface.co/google/flan-t5-xxl/discussions/41
            # regarding max_length: https://github.com/google-research/FLAN/issues/36
            input_ids = self.lm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            labels = self.lm_tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
            prompt_ans_lm_inputs = {"input_ids": input_ids, "labels": labels}
        else:
            if "Llama-2" in self.lm_name:
                max_length = 4096
            elif "llama-" in self.lm_name: # llama 1
                max_length = 2048
            elif "gpt2" in self.lm_name:
                max_length = 1024
            else:
                max_length = 256
            prompt_ans_lm_inputs = self.lm_tokenizer(
                prompt, answer, max_length=max_length, padding=True, truncation=True, 
                return_tensors='pt', return_token_type_ids=True
            )

        return {
            "query_inputs": query_inputs,
            "doc_embeddings": doc_embeddings,
            "prompt_ans_lm_inputs": prompt_ans_lm_inputs,
            "num_has_answer": num_has_answer,
        }
    
def validate(
        query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
        accelerator, model_max_length, train_step_logdir
):
    logger.info("*** Start validation ***")
    query_encoder.eval()
    language_model.eval()
    total_loss = 0
    total_ans_prob = 0
    num_batches = 0
    all_retriever_pick = []
    all_predictions = []
    total_num_correct = 0
    total_num_examples = 0
    total_too_long = 0
    total_has_answer = 0

    for step, batch in tqdm(enumerate(dev_dataloader)):
        with torch.no_grad():
            ## Metric 1. Loss
            logger.debug("...Sending batch to model...")
            logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
            logger.debug(f"batch['doc_embeddings']: {batch['doc_embeddings'].shape}")
            logger.debug(f"query_encoder device: {next(query_encoder.parameters()).device}")
            query_embedding = query_encoder(**batch['query_inputs']).pooler_output \
                if "dpr" in args.query_encoder \
                else query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:]
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

            query_embedding, doc_embedding = query_embedding.to("cpu"), doc_embedding.to("cpu")
            del query_embedding, doc_embedding
            torch.cuda.empty_cache()
            logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            logger.debug("...Calculating loss from LM...")
            if "flan" in args.lm_model:
                lm_score = get_t5_lm_score(
                    **batch['prompt_ans_lm_inputs'],
                    model=language_model,
                    device=accelerator.device,
                    tokenizer=lm_tokenizer,
                    max_length=model_max_length,
                    max_tokens_to_generate=args.max_tokens_to_generate,
                    num_orig_question=num_orig_question,
                    llm_batch_size=args.llm_batch_size,
                )
            else:
                lm_score = get_lm_score(
                    **batch['prompt_ans_lm_inputs'],
                    model=language_model,
                    device=accelerator.device,
                    max_length=model_max_length,
                    max_tokens_to_generate=args.max_tokens_to_generate,
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
            # TODO weird... 又好像沒問題...
            total_ans_prob += lm_score.exp().mean().item()
            all_retriever_pick.extend(retrievers_pick.tolist())
            retriever_score, lm_score = retriever_score.to("cpu"), lm_score.to("cpu")
            del retriever_score, lm_score
            torch.cuda.empty_cache()

            ## Metric 3. Exact match
            # reshape from [n_question*n_comb,seq_len] to [n_question,n_comb,seq_len]
            batch['prompt_ans_lm_inputs'] = {
                k: v.view(num_orig_question, -1, v.shape[-1]) for k,v in batch['prompt_ans_lm_inputs'].items()
            }
            # and then take the retriever's pick for each question
            batch['prompt_ans_lm_inputs'] = {
                k: v[torch.arange(num_orig_question),retrievers_pick] for k,v in batch['prompt_ans_lm_inputs'].items()
            }
            batch_result = lm_gen_and_check(
                model=language_model, 
                tokenizer=lm_tokenizer,
                device=accelerator.device,
                max_length=model_max_length,
                prompt_ans_lm_inputs=batch['prompt_ans_lm_inputs'],
                max_tokens_to_generate=args.max_tokens_to_generate,
                train_step_logdir=train_step_logdir,
                llm_batch_size=args.llm_batch_size,
            )
            total_num_correct += batch_result["num_correct"]
            total_num_examples += batch_result["num_examples"]
            total_too_long += batch_result["too_long"]
            all_predictions.extend(batch_result["predictions"])

            num_batches += 1
            total_has_answer += batch["num_has_answer"]

    # write retriever pick to train_step_logdir
    with open(os.path.join(train_step_logdir, "retriever_pick.txt"), "w") as f:
        for pick in all_retriever_pick:
            f.write(str(pick) + "\n")
    with open(os.path.join(train_step_logdir, "prediction.json"), "w", encoding='utf-8') as f:
        for item in all_predictions:
            f.write(item + "\n")

    avg_loss = total_loss / num_batches
    avg_prob = total_ans_prob / num_batches
    exact_match = total_num_correct / total_num_examples * 100
    final_result = {"loss": avg_loss, "avg_prob": avg_prob, "exact_match": exact_match, \
            "too_long": total_too_long, "has_answer": total_has_answer}
    for k,v in final_result.items():
        logger.info(f"{k}: {v}")
    return final_result


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

    accelerator.init_trackers(
        project_name="dpr", 
        config=args,
        init_kwargs={"wandb":{"name":f"({args.data_size}) {model_short_name}-{args.max_round}round-{args.k}k-bs({args.per_device_train_batch_size}&{args.per_device_eval_batch_size})({args.llm_batch_size})"}}
    )
    # %%
    if not debug and accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir
        wandb_tracker.run.log_code(".")
        wandb_tracker.run.tags = [
            f"data_size: {args.data_size}", f"language_model: {args.lm_model}", 
            f"query_encoder: {args.query_encoder}", f"doc_encoder: {args.doc_encoder}", 
            f"max_round: {args.max_round}", f"k: {args.k}", 
            f"train_bs: {args.per_device_train_batch_size}", f"eval_bs: {args.per_device_eval_batch_size}",
            "doc->question", "train", "sep:/n"
        ]
    else:
        LOG_DIR = "./tmp_log"  # Or any other directory you want to use when debugging
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
    # skip those exemplars
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
    train_dataset = QADataset(train_qa_pairs, train_corpus, train_doc_embeddings, query_tokenizer, lm_tokenizer, query_encoder, 'train', args, accelerator)
    dev_dataset = QADataset(dev_qa_pairs, dev_corpus, dev_doc_embeddings, query_tokenizer, lm_tokenizer, query_encoder, 'dev', args, accelerator)
    
    logger.info("...Deleting train_data and dev_data...")
    del train_data, dev_data

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
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)
    completed_steps = 0

    # TODO restore after debug
    logger.info(f"\n...0 Step Evaluation...")
    steps_log_dir = os.path.join(LOG_DIR,f"step-{completed_steps}")
    if not os.path.exists(steps_log_dir):
        os.makedirs(steps_log_dir)
    loss = validate(query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, steps_log_dir)
    accelerator.log({"eval":loss}, step=completed_steps)

    logger.info("\n***** Running training *****")
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
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    start_time = time.time()

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
                    # logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
                    # logger.debug(f"batch['doc_embeddings']: {batch['doc_embeddings'].shape}")
                    query_embedding = query_encoder(**batch['query_inputs']).pooler_output \
                        if "dpr" in args.query_encoder \
                        else query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:]
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

                    del query_embedding, doc_embedding  
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                    logger.debug(f"...Calculating loss from LM on {accelerator.device}...")
                    # very likely to OOM error here
                    if "t5" in args.lm_model:
                        lm_score = get_t5_lm_score(
                            **batch['prompt_ans_lm_inputs'],
                            model=language_model,
                            device=accelerator.device,
                            tokenizer=lm_tokenizer,
                            max_length=model_max_length,
                            max_tokens_to_generate=args.max_tokens_to_generate,
                            num_orig_question=num_orig_question,
                            llm_batch_size=args.llm_batch_size,
                        )
                    else:
                        lm_score = get_lm_score(
                            **batch['prompt_ans_lm_inputs'],
                            model=language_model,
                            device=accelerator.device,
                            max_length=model_max_length,
                            max_tokens_to_generate=args.max_tokens_to_generate,
                            num_orig_question=num_orig_question,
                            llm_batch_size=args.llm_batch_size,
                        )
                    
                    # move retriever_score back to GPU
                    retriever_score = retriever_score.to(accelerator.device)
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                    logger.debug(f"retriever_score.device: {retriever_score.device}; lm_score.device: {lm_score.device}")

                    # # try to fix RuntimeError: Found dtype Float but expected Half
                    # for param in query_encoder.parameters():
                    #     # Check if parameter dtype is  Float (float32)
                    #     if param.dtype == torch.float32:
                    #         param.data = param.data.to(torch.float16)
                    # try to fix RuntimeError: Found dtype Float but expected Half
                    retriever_score = retriever_score.half()
                    lm_score = lm_score.half()

                    loss = calculate_KL_div_loss(input_logits=retriever_score, target_logits=lm_score, temperature=args.temperature)
                    retriever_score, lm_score = retriever_score.to("cpu"), lm_score.to("cpu")
                    del retriever_score, lm_score
                    logger.debug(f"Finish compute loss. loss = {loss}")
                    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                # fix llama error
                # RuntimeError: Found dtype Float but expected Half
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
                        logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                        logger.info(f"\n...Evaluation...")
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
                            query_tokenizer.save_pretrained(query_encoder_path)
                            logger.info(f"Checkpoint saved to {LOG_DIR}")
                            
                        accelerator.wait_for_everyone()
                
                optimizer.step()
                optimizer.zero_grad()
                logger.debug(f"Finish step {step}.\n GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    
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