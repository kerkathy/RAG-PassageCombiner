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
    get_lm_prob,
    get_t5_lm_prob,
    lm_gen_and_check,
    load_doc_encoder_and_tokenizer,
    load_query_encoder_and_tokenizer,
    make_prompt,
    text_has_answer,
)

# def calculate_dpr_loss(matching_score,labels):
#     return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

def calculate_KL_div_loss(
    input_logits, # [n_question,n_comb]
    target_logits, # [n_question,n_comb]
    temperature,
):
    """
    Calculate KL divergence loss between input and target logits
    Note: input_logits and target_logits are logits, not distributions
    """
    global logger
    # logger.debug(f"input_logits: {F.softmax(input_logits / temperature, dim=1)}")
    # logger.debug(f"target_logits: {F.softmax(target_logits / temperature, dim=1)}")
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    loss = kl_loss(
        F.log_softmax(input_logits / temperature, dim=1), # input should be a distribution in the log space
        F.softmax(target_logits / temperature, dim=1),
    )
    return loss

# def inloop_getitem
def inloop_extend_item(data, corpus, doc_embeddings, ret_tokenizer, query_encoder, args):
    global logger
    # logger.debug("In inloop_getitem...")
    embedding_device = data[0][3].device

    # Initialize pointers
    cur_visited = 0
    this_round_should_visited = 0
    next_round_should_visited = len(data)

    for i_rnd in range(args.max_round):
        # logger.debug(f"Round {i_rnd} has {next_round_should_visited} data to go thru...")
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
            # logger.debug(f"query: {doc_list[-1] + ' ' + query}")
            doc_ids = retrieve_top_k_docid(doc_list[-1] + " " + query, doc_embeddings, ret_tokenizer, query_encoder, args.k)
            # Append new data
            for docid in doc_ids:
                new_doc_list = doc_list + [corpus[docid]]
                data.append((query, new_doc_list, answer, doc_embeddings[docid].to(embedding_device)))

                # Increment next_pointer
                next_round_should_visited += 1
    # logger.debug(f"After getitem, data size: {len(data)}")
    return data  # List of tuples


def inloop_collate_fn(samples, ret_tokenizer, lm_tokenizer, lm_name, args):
    """
    samples: List[List[tuple]]
    """
    global logger
    # TODO add feature: 不同文章數量的分開 decode
    # flatten the samples into a list of tuples
    # logger.debug(f"Original batch size: {len(samples)}")
    samples = [item for sublist in samples for item in sublist]
    # logger.debug(f"Real batch size: {len(samples)}")
    
    # each item is (query, all_doc, answer, last_doc_embedding)
    query_inputs = ret_tokenizer([x[0] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
    # collect doc_inputs from doc_embeddings
    doc_embeddings = torch.stack([x[3] for x in samples], dim=0)
    
    # in each item, num_docs shuold be num items where x[1][i] != "" (empty doc holder)
    prompt = [make_prompt(
        question=x[0], documents=x[1], lm_name=lm_name, 
        num_docs=len([doc for doc in x[1] if doc != ""]),
        num_exemplars=args.num_exemplars, dataset=args.dataset_name) for x in samples]
    answer = [x[2] for x in samples]
    num_has_answer = 0
    for a, p in zip(answer, prompt):
        # logger.debug(f"Answer: {a}")
        # logger.debug(f"Prompt: {p}")
        if text_has_answer(a, p):
            num_has_answer += 1

    if "t5" in lm_name:
        # separate input_ids (send into encoder) and labels (send into decoder)
        # regarding max_length: https://huggingface.co/google/flan-t5-xxl/discussions/41
        # regarding max_length: https://github.com/google-research/FLAN/issues/36
        input_ids = lm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        labels = lm_tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        prompt_ans_lm_inputs = {"input_ids": input_ids, "labels": labels}
    else:
        if "Llama-2" in lm_name:
            max_length = 4096
        elif "llama-" in lm_name: # llama 1
            max_length = 2048
        elif "gpt2" in lm_name:
            max_length = 1024
        else:
            max_length = 256
        prompt_ans_lm_inputs = lm_tokenizer(
            prompt, answer, max_length=max_length, padding=True, truncation=True, 
            return_tensors='pt', return_token_type_ids=True
        )
    
    # update max token length
    global max_ret_token_len, max_lm_token_len
    max_ret_token_len = max(max_ret_token_len, query_inputs["input_ids"].shape[1])
    max_lm_token_len = max(max_lm_token_len, prompt_ans_lm_inputs["input_ids"].shape[1])

    return {
        "query_inputs": query_inputs, # dict
        "doc_embeddings": doc_embeddings, # tensor, [bs,n_dim]
        "prompt_ans_lm_inputs": prompt_ans_lm_inputs, # dict
        "num_has_answer": num_has_answer, # int
    }
    

def validate(
        query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
        accelerator, model_max_length, train_step_logdir, logger
):
    logger.info(f"*** Start validation at {train_step_logdir.split('/')[-1]} ***")
    query_encoder.eval()
    language_model.eval()
    total_loss = 0
    total_ans_prob = 0
    num_batches = len(dev_dataloader)
    all_retriever_pick = []
    all_predictions = []
    total_num_correct = 0
    total_num_examples = 0
    total_too_long = 0
    total_has_answer = 0
    total_num_correct_pick = 0
    total_retrievers_pick_lm_prob = 0

    for step, raw_batch in tqdm(enumerate(dev_dataloader)):
        # make raw_batch into a extened batch
        # by first extend each item and then collate_fn
        with torch.no_grad():
            extended_batch = [inloop_extend_item(
                data=x["data"], corpus=x["corpus"], doc_embeddings=x["doc_embeddings"],
                ret_tokenizer=query_tokenizer, query_encoder=query_encoder, args=args
            ) for x in raw_batch]
        batch = inloop_collate_fn(
            samples=extended_batch, ret_tokenizer=query_tokenizer, lm_tokenizer=lm_tokenizer, 
            lm_name=args.lm_model, args=args
        )
        del extended_batch, raw_batch
        
        batch["doc_embeddings"] = batch["doc_embeddings"].to(accelerator.device)
        batch["query_inputs"] = {k: v.to(accelerator.device) for k,v in batch["query_inputs"].items()}
        batch["prompt_ans_lm_inputs"] = {k: v.to(accelerator.device) for k,v in batch["prompt_ans_lm_inputs"].items()}

        logger.info(f"[validation step {step}/{num_batches}] max_ret_token_len: {batch['query_inputs']['input_ids'].shape[1]}")
        logger.info(f"[validation step {step}/{num_batches}] max_lm_token_len: {batch['prompt_ans_lm_inputs']['input_ids'].shape[1]}")
        
        with torch.no_grad():
            ## Metric 1. Loss
            query_embedding = query_encoder(**batch['query_inputs']).pooler_output \
                if "dpr" in args.query_encoder \
                else query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:] # [bs,n_dim]
            doc_embedding = batch["doc_embeddings"] # 
            
            # logger.debug(f"query_embedding device: {query_embedding.device}; doc_embedding device: {doc_embedding.device}")
            logger.info(f"[Sent to query encoder] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            
            single_device_query_num, _ = query_embedding.shape
            single_device_doc_num = doc_embedding.shape[0]

            logger.info("...Waiting for everyone...")
            if accelerator.use_distributed:
                doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                doc_list[dist.get_rank()] = doc_embedding
                doc_embedding = torch.cat(doc_list, dim=0)

                query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                query_list[dist.get_rank()] = query_embedding
                query_embedding = torch.cat(query_list, dim=0)

            # convert query_embedding and doc_embedding to unit vectors
            query_embedding = F.normalize(query_embedding, p=2, dim=1) # p: norm type
            doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
            retriever_cossim = torch.sum(query_embedding * doc_embedding, dim=1)  # [bs]
            num_orig_question = single_device_query_num // sum([args.k ** i for i in range(args.max_round + 1)])
            retriever_cossim = retriever_cossim.reshape(num_orig_question, -1)
            logger.info(f"[Got ret cos sim] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            query_embedding, doc_embedding = query_embedding.to("cpu"), doc_embedding.to("cpu")
            del query_embedding, doc_embedding
            torch.cuda.empty_cache()
            logger.info(f"[Emptied embedding cache] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            if "flan" in args.lm_model:
                lm_prob = get_t5_lm_prob(
                    **batch['prompt_ans_lm_inputs'],
                    model=language_model,
                    device=accelerator.device,
                    tokenizer=lm_tokenizer,
                    max_length=model_max_length,
                    max_tokens_to_generate=args.max_tokens_to_generate,
                    num_orig_question=num_orig_question,
                    llm_batch_size=args.eval_llm_batch_size,
                    logger=logger,
                )
            else:
                lm_prob = get_lm_prob(
                    **batch['prompt_ans_lm_inputs'],
                    model=language_model,
                    device=accelerator.device,
                    max_length=model_max_length,
                    max_tokens_to_generate=args.max_tokens_to_generate,
                    num_orig_question=num_orig_question,
                    llm_batch_size=args.eval_llm_batch_size,
                    logger=logger,
                )
            logger.info(f"[Got LM prob] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            
            loss = calculate_KL_div_loss(input_logits=retriever_cossim, target_logits=lm_prob, temperature=args.temperature)
            logger.info(f"[Got KL loss] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            ## Metric 2. Average answer probability
            # for each question, take idx of max retriever_cossim 
            # get its corresponding lm_prob
            # note. retriever_cossim and lm_prob are both [n_question,n_comb]
            retrievers_pick = torch.argmax(retriever_cossim,dim=1) # [n_question]

            # ### debug
            # print(f"retriever_cossim: {retriever_cossim}")
            # print(f"softmax retriever score: {F.softmax(retriever_cossim / temperature,dim=1)}")
            # print(f"lm_prob: {lm_prob}")
            # print(f"softmax lm score: {F.softmax(lm_prob / temperature / temperature, dim=1)}")
            # print(f"retrievers_pick: {retrievers_pick}")
            # print(f"lm score each question max: {lm_prob[torch.arange(num_orig_question),torch.argmax(lm_prob,dim=1)]}")
            # print(f"retrievers pick lm score: {lm_prob[torch.arange(num_orig_question),retrievers_pick]}")
            # ### debug

            total_num_correct_pick += (retrievers_pick == torch.argmax(lm_prob,dim=1)).sum().item()
            lm_prob = lm_prob[torch.arange(num_orig_question),retrievers_pick] # [n_question]
            # total_ans_prob += lm_prob.exp().sum().item() 
            total_ans_prob += lm_prob.sum().item() 
            # count how many retriever's pick is the same as lm's pick
            all_retriever_pick.extend(retrievers_pick.tolist())
            retriever_cossim, lm_prob = retriever_cossim.to("cpu"), lm_prob.to("cpu")
            del retriever_cossim, lm_prob
            torch.cuda.empty_cache()
            logger.info(f"[Emptied scoring cache] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            # ## Metric 3. Exact match
            for k,v in batch['prompt_ans_lm_inputs'].items():
                v = v.view(num_orig_question, -1, v.shape[-1])[torch.arange(num_orig_question),retrievers_pick]

            batch_result = lm_gen_and_check(
                model=language_model, 
                tokenizer=lm_tokenizer,
                device=accelerator.device,
                max_length=model_max_length,
                prompt_ans_lm_inputs=batch['prompt_ans_lm_inputs'],
                accelerator=accelerator,
                max_tokens_to_generate=args.max_tokens_to_generate,
                llm_batch_size=args.eval_llm_batch_size,
                logger=logger,
            )
            total_num_correct += batch_result["num_correct"]
            total_num_examples += batch_result["num_examples"]
            total_too_long += batch_result["too_long"]
            all_predictions.extend(batch_result["predictions"])
            total_loss += loss.item() * batch_result["num_examples"]

            # num_batches += 1
            total_has_answer += batch["num_has_answer"]

    # write retriever pick to train_step_logdir
    with open(os.path.join(train_step_logdir, "retriever_pick.txt"), "w") as f:
        for pick in all_retriever_pick:
            f.write(str(pick) + "\n")
    with open(os.path.join(train_step_logdir, "prediction.json"), "w", encoding='utf-8') as f:
        for item in all_predictions:
            f.write(item + "\n")

    final_result = {
        "avg_loss": total_loss / total_num_examples, 
        "avg_prob": total_ans_prob / total_num_examples, # 這裡原本算錯啦! 原本是每個 batch 的 mean 加起來再除以 num_batches
        "exact_match (%)": total_num_correct / total_num_examples * 100,
        "too_long (%)": total_too_long / total_num_examples * 100,
        "has_answer (%)": total_has_answer / total_num_examples * 100,
        "retriever_pick_acc (%)": total_num_correct_pick / total_num_examples * 100,
        "retriever_pick_lm_prob": total_retrievers_pick_lm_prob / total_num_examples,
    }
    logger.info(f"Done {train_step_logdir.split('/')[-1]} step validation.")
    for k,v in final_result.items():
        logger.info(f"{k}: {v}")
    return final_result


def train_and_eval(
        query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
        accelerator, model_max_length, optimizer, lr_scheduler, logger,
        step, epoch, completed_steps, MAX_TRAIN_STEPS, EVAL_STEPS, LOG_DIR, progress_bar,
):
    """
    Train the model and evaluate it at every `EVAL_STEPS` steps.
    """
    # make raw_batch into a extened batch
    # by first extend each item and then collate_fn
    with torch.no_grad():
        extended_batch = [inloop_extend_item(
            data=x["data"], corpus=x["corpus"], doc_embeddings=x["doc_embeddings"],
            ret_tokenizer=query_tokenizer, query_encoder=query_encoder, args=args
        ) for x in raw_batch]
    batch = inloop_collate_fn(
        samples=extended_batch, ret_tokenizer=query_tokenizer, lm_tokenizer=lm_tokenizer, 
        lm_name=args.lm_model, args=args
    )
    
    batch["doc_embeddings"] = batch["doc_embeddings"].to(accelerator.device)
    batch["query_inputs"] = {k: v.to(accelerator.device) for k,v in batch["query_inputs"].items()}
    batch["prompt_ans_lm_inputs"] = {k: v.to(accelerator.device) for k,v in batch["prompt_ans_lm_inputs"].items()}

    # print max input seq len in this batch
    logger.info(f"[train step {step} (globally {completed_steps})] max_ret_token_len: {batch['query_inputs']['input_ids'].shape[1]}")
    logger.info(f"[train step {step} (globally {completed_steps})] max_lm_token_len: {batch['prompt_ans_lm_inputs']['input_ids'].shape[1]}")
    del extended_batch, raw_batch

    query_encoder.train()
    with accelerator.accumulate(query_encoder): # gradient accumulation
        with accelerator.autocast():
            # logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
            # logger.debug(f"batch['doc_embeddings']: {batch['doc_embeddings'].shape}")
            query_embedding = query_encoder(**batch['query_inputs']).pooler_output \
                if "dpr" in args.query_encoder \
                else query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:]
            doc_embedding = batch["doc_embeddings"]
            logger.info(f"[Sent to query encoder] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            
            # shape of both query_embedding and doc_embedding: [bs,n_dim]
            # where bs = n_comb * num_orig_question
            single_device_query_num,_ = query_embedding.shape
            single_device_doc_num = doc_embedding.shape[0]

            logger.info("...Waiting for everyone...")
            if accelerator.use_distributed:
                doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                doc_list[dist.get_rank()] = doc_embedding
                doc_embedding = torch.cat(doc_list, dim=0)

                query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                query_list[dist.get_rank()] = query_embedding
                query_embedding = torch.cat(query_list, dim=0)

            query_embedding = F.normalize(query_embedding, p=2, dim=1) # p: norm type
            doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
            retriever_cossim = torch.sum(query_embedding * doc_embedding, dim=1)  # [bs]
            num_orig_question = single_device_query_num // sum([args.k ** i for i in range(args.max_round + 1)])
            retriever_cossim = retriever_cossim.reshape(num_orig_question, -1)
            logger.info(f"[Got ret cos sim] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")

            query_embedding, doc_embedding = query_embedding.to("cpu"), doc_embedding.to("cpu")
            del query_embedding, doc_embedding
            torch.cuda.empty_cache()
            logger.info(f"[Emptied embedding cache] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            # very likely to OOM error here
            if "t5" in args.lm_model:
                lm_prob = get_t5_lm_prob(
                    **batch['prompt_ans_lm_inputs'],
                    model=language_model,
                    device=accelerator.device,
                    tokenizer=lm_tokenizer,
                    max_length=model_max_length,
                    max_tokens_to_generate=args.max_tokens_to_generate,
                    num_orig_question=num_orig_question,
                    llm_batch_size=args.train_llm_batch_size,
                    logger=logger,
                )
            else:
                lm_prob = get_lm_prob(
                    **batch['prompt_ans_lm_inputs'],
                    model=language_model,
                    device=accelerator.device,
                    max_length=model_max_length,
                    max_tokens_to_generate=args.max_tokens_to_generate,
                    num_orig_question=num_orig_question,
                    llm_batch_size=args.train_llm_batch_size,
                    logger=logger,
                )
            logger.info(f"[Got LM prob] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
            
            # move retriever_cossim back to GPU
            # retriever_cossim = retriever_cossim.to(accelerator.device)
            # logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            # logger.debug(f"retriever_cossim.device: {retriever_cossim.device}; lm_prob.device: {lm_prob.device}")

            # # try to fix RuntimeError: Found dtype Float but expected Half
            # for param in query_encoder.parameters():
            #     # Check if parameter dtype is  Float (float32)
            #     if param.dtype == torch.float32:
            #         param.data = param.data.to(torch.float16)
            # try to fix RuntimeError: Found dtype Float but expected Half
            retriever_cossim = retriever_cossim.half()
            lm_prob = lm_prob.half()

            loss = calculate_KL_div_loss(input_logits=retriever_cossim, target_logits=lm_prob, temperature=args.temperature)
            retriever_cossim, lm_prob = retriever_cossim.to("cpu"), lm_prob.to("cpu")
            del retriever_cossim, lm_prob
            logger.info(f"[Got KL loss] loss = {loss}; GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")

        # fix llama error
        # RuntimeError: Found dtype Float but expected Half
        accelerator.backward(loss)
        logger.info(f"[After backward] loss = {loss}; GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")

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
                logger.info(f"[Before evaluation] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB.  Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
                logger.info(f"[Before evaluation] max ret token len: {max_ret_token_len}; max lm token len: {max_lm_token_len}")
                steps_log_dir = os.path.join(LOG_DIR,f"step-{completed_steps}")
                if not os.path.exists(steps_log_dir):
                    os.makedirs(steps_log_dir)
                loss = validate(query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, steps_log_dir, logger)
                query_encoder.train() # Make sure the model is back in training mode after validation
                accelerator.log({"eval":loss}, step=completed_steps)
                accelerator.wait_for_everyone()
                logger.info(f"[Got every eval subproc] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                if accelerator.is_local_main_process:
                    # query_encoder_path = os.path.join(steps_log_dir, "query_encoder")
                    ckpt_path = os.path.join(steps_log_dir, "checkpoint.pt")
                    ensure_directory_exists_for_file(ckpt_path)
                    # unwrap the model from DDP
                    unwrapped_model = accelerator.unwrap_model(query_encoder)
                    unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                    torch.save({
                        'query_encoder': unwrapped_model.state_dict(),
                        'optimizer': unwrapped_optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'completed_steps': completed_steps,}, ckpt_path)
                    # query_encoder.save_pretrained(query_encoder_path)
                    # query_tokenizer.save_pretrained(query_encoder_path)
                    logger.info(f"Checkpoint saved to {ckpt_path}")
                    
                accelerator.wait_for_everyone()
        
        optimizer.step()
        optimizer.zero_grad()
        logger.info(f"[Finish step {step} in epoch {epoch} (globally {completed_steps})] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB.  Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")