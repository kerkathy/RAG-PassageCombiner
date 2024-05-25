# %%
## built-in
import time,random,queue,sys
import math,logging,json,random,os,psutil
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud
os.environ["CUDA_LAUNCH_BLOCKING"]="1" ## for debugging
import gc

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
)

debug = True # set log mode to debug, and stop wandb logging
max_ret_token_len = 0
max_lm_token_len = 0

logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
logger = get_logger(__name__)

def parse_args():
    # # When using ipynb
    # config_file = 'config/24G_train_dpr_nq.yaml'
    # yaml_config = get_yaml_file(config_file)
    # args_dict = {}
    # args_dict['config_file'] = config_file

    # When using cmd line
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/24G_train_dpr_nq.yaml')
    args = parser.parse_args()
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config = get_yaml_file(args.config_file)

    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config) # access in attribute style
    return args

# def calculate_dpr_loss(matching_score,labels):
#     return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

def calculate_KL_div_loss(
    input_logits, # size [n_question,n_comb]
    target_logits, # size [n_question,n_comb]
    temperature, # [ret_temperature,lm_temperature]
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
        F.log_softmax(input_logits / temperature[0], dim=1), # input should be a distribution in the log space
        F.softmax(target_logits / temperature[1], dim=1),
    )
    return loss

def calculate_cross_entropy_loss(
    input_logits, # [n_question,n_comb]
    target_logits, # [n_question,n_comb]
    temperature, # [ret_temperature,lm_temperature]
):
    """
    Calculate cross entropy loss between input and target logits
    Take the argmax of target_logits as the label
    """
    global logger
    # logger.debug(f"input_logits: {F.softmax(input_logits / temperature, dim=1)}")
    # logger.debug(f"target_logits: {F.softmax(target_logits / temperature, dim=1)}")
    ce_loss = nn.CrossEntropyLoss() # reduction is mean by default
    input_logits = input_logits / temperature[0]
    loss = ce_loss(
        input=input_logits, # input is expected to contain the unnormalized logits for each class
        target=torch.argmax(target_logits, dim=1),
    )
    return loss 

def calculate_nll_loss(
    doc_scores, # [n_question,n_comb]
    seq_probs, # [n_question,n_comb]
):
    """
    Following RAG paper, calculate negative log likelihood loss
    prob_y_given_x = doc similarity score * answer probability
    NLL = -log(prob_y_given_x)
    """
    # ref: https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/rag/modeling_rag.py#L1057
    doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1)
    seq_logprobs = seq_probs.log()
    nll_loss = -(doc_logprobs + seq_logprobs).logsumexp(dim=1).mean()
    return nll_loss

class QADataset(torch.utils.data.Dataset):
    def __init__(self, qa_pairs, all_corpus, all_doc_embeddings):
        self.qa_pairs = qa_pairs
        self.all_corpus = all_corpus
        self.all_doc_embeddings = all_doc_embeddings
        
    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        data = [self.qa_pairs[idx]]  # each item is (query, all_doc, answer, last_doc_embedding, qid)
        corpus = self.all_corpus[idx]
        doc_embeddings = self.all_doc_embeddings[idx]  # Move to correct device
        # debug
        # print(f"{idx}th data len(corpus): {len(corpus)}")
        # print(f"{idx}th data doc_embeddings.shape: {doc_embeddings.shape}")
        return {"data": data, "corpus": corpus, "doc_embeddings": doc_embeddings}
    
    def collate_fn(self, samples):
        """
        samples: List[Dict]
        """
        return samples

# this is like getitem, moved outside Dataset because we're using GPU here, and using GPU inside Dataset is not recommended
# def inloop_getitem
def inloop_extend_item(data, corpus, doc_embeddings, ret_tokenizer, query_encoder, args):
    """
    Extend each item in data by retrieving top k documents for each round
    into 1 + k + k^2 + ... + k^max_round items
    data: List[tuple], each tuple is (query, all_doc, answer, last_doc_embedding)
    """
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
            with torch.no_grad():
                doc_ids = retrieve_top_k_docid(doc_list[-1] + " " + query, doc_embeddings, ret_tokenizer, query_encoder, args.k)
            # Append new data
            for docid in doc_ids:
                new_doc_list = doc_list + [corpus[docid]]
                data.append((query, new_doc_list, answer, doc_embeddings[docid].to(embedding_device)))

                # Increment next_pointer
                next_round_should_visited += 1

    # debug: temporarily remove empty document
    if not args.empty_doc:
        num_data_before_remove = len(data)
        data = [x for x in data if x[1] != [""]]
        num_data_after_remove = len(data)
        assert num_data_before_remove == num_data_after_remove + 1, f"num_data_before_remove ({num_data_before_remove}) != num_data_after_remove + 1 ({num_data_after_remove + 1})"

    return data  # List of tuples

# %%
# this is like collate_fn, moved outside Dataset because getitem and collate_fn should be in the same scope
def inloop_collate_fn(samples, ret_tokenizer, lm_tokenizer, lm_name, args, mode="train"):
    """
    Construct a batch.
    samples: List[List[tuple]]
    """
    global logger
    # TODO add feature: 不同文章數量的分開 decode
    # flatten the samples into a list of tuples
    # logger.debug(f"Original batch size: {len(samples)}")
    num_orig_question = len(samples)
    samples = [item for sublist in samples for item in sublist]
    # logger.debug(f"Real batch size: {len(samples)}")
    
    # each item is (query, all_doc, answer, last_doc_embedding)
    query_inputs = ret_tokenizer([x[0] for x in samples], max_length=256, padding=True, truncation=True, return_tensors='pt')
    # collect doc_inputs from doc_embeddings
    doc_embeddings = torch.stack([x[3] for x in samples], dim=0)
    
    prompt = [make_prompt(
        question=x[0], documents=x[1], lm_name=lm_name, 
        num_exemplars=args.num_exemplars, dataset=args.dataset_name) for x in samples]
    answer_to_encode = [x[2][0] for x in samples] # pick the first answer for each question, as eval set may have multiple answers
    # debug
    # print(f"Real answer: {answer}")

    if "t5" in lm_name:
        # separate input_ids (send into encoder) and labels (send into decoder)
        # regarding max_length: https://huggingface.co/google/flan-t5-xxl/discussions/41
        # regarding max_length: https://github.com/google-research/FLAN/issues/36
        input_ids = lm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        labels = lm_tokenizer(answer_to_encode, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        prompt_ans_lm_inputs = {"input_ids": input_ids, "labels": labels}
    else:
        if "Llama-3" in lm_name:
            max_length = 8000
            lm_tokenizer.pad_token_id = lm_tokenizer.eos_token_id
        if "Llama-2" in lm_name:
            max_length = 4096
        elif "llama-" in lm_name: # llama 1
            max_length = 2048
        elif "gpt2" in lm_name:
            max_length = 1024
        else:
            max_length = 256
        
        # add_special_tokens=False is really important as it affects probability A LOT!
        # those bos, eos are already added in make_prompt for llama3
        prompt_ans_lm_inputs = lm_tokenizer(
            prompt, answer_to_encode, max_length=max_length, padding=True, truncation=True, 
            return_tensors='pt', return_token_type_ids=True,
            add_special_tokens=False if "Llama-3" in lm_name else True,
        ) # dict of keys: input_ids, attention_mask, token_type_ids. Each of shape [num_orig_question * n_comb, n_dim]
    
    # update max token length
    global max_ret_token_len, max_lm_token_len
    max_ret_token_len = max(max_ret_token_len, query_inputs["input_ids"].shape[1])
    max_lm_token_len = max(max_lm_token_len, prompt_ans_lm_inputs["input_ids"].shape[1])

    res_dict = {
        "query_inputs": query_inputs, # dict
        "doc_embeddings": doc_embeddings, # tensor, [bs,n_dim]
        "prompt_ans_lm_inputs": prompt_ans_lm_inputs, # dict
    }
    if mode == "eval":
        n_comb = prompt_ans_lm_inputs["input_ids"].shape[0] // num_orig_question
        res_dict["full_answers"] = [x[2] for i, x in enumerate(samples) if i % n_comb == 0] # list of list of str; len = num_orig_question
        assert len(res_dict["full_answers"]) == num_orig_question, f"len(res_dict['full_answers']) ({len(res_dict['full_answers'])}) != num_orig_question ({num_orig_question})"
        if "llama" in lm_name.lower():
            res_dict["prompt_strs"] = prompt # list[str], len = num_orig_question * n_comb
    return res_dict

# %%
def validate(
        query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, 
        accelerator, model_max_length, train_step_logdir
):
    # %%
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
    total_f1_score = 0

    # %%
    for step, raw_batch in tqdm(enumerate(dev_dataloader)):
        # %%
        # make raw_batch into a extened batch by first extend each item and then collate_fn
        extended_batch = [inloop_extend_item(
            data=x["data"], corpus=x["corpus"], doc_embeddings=x["doc_embeddings"],
            ret_tokenizer=query_tokenizer, query_encoder=query_encoder, args=args
        ) for x in raw_batch]
        batch = inloop_collate_fn(
            samples=extended_batch, ret_tokenizer=query_tokenizer, lm_tokenizer=lm_tokenizer, 
            lm_name=args.lm_model, args=args, mode="eval"
        ) # dict of keys: query_inputs, doc_embeddings, prompt_ans_lm_inputs, full_answers, [prompt_strs]
        del extended_batch, raw_batch
        
        batch["doc_embeddings"] = batch["doc_embeddings"].to(accelerator.device)
        batch["query_inputs"] = {k: v.to(accelerator.device) for k,v in batch["query_inputs"].items()}
        batch["prompt_ans_lm_inputs"] = {k: v.to(accelerator.device) for k,v in batch["prompt_ans_lm_inputs"].items()}
    
        logger.info(f"[validation step {step}/{num_batches}] max_ret_token_len: {batch['query_inputs']['input_ids'].shape[1]}")
        logger.info(f"[validation step {step}/{num_batches}] max_lm_token_len: {batch['prompt_ans_lm_inputs']['input_ids'].shape[1]}")
        
        # %%
        with torch.no_grad():
            ## Metric 1. Loss
            query_embedding = query_encoder(**batch['query_inputs']).pooler_output \
                if "dpr" in args.query_encoder \
                else query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:] # [bs,n_dim]
            doc_embedding = batch["doc_embeddings"]
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
            num_orig_question = single_device_query_num // sum([args.k ** i for i in range(args.max_round + 1)]) if args.empty_doc \
                else single_device_query_num // (sum([args.k ** i for i in range(args.max_round + 1)]) - 1)
            retriever_cossim = retriever_cossim.view(num_orig_question, -1)
            logger.info(f"[Got ret cos sim] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            query_embedding, doc_embedding = query_embedding.to("cpu"), doc_embedding.to("cpu")
            del query_embedding, doc_embedding
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"[Emptied embedding cache] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            # %%
            if "t5" in args.lm_model:
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
                    # tokenizer=lm_tokenizer, # for debugging only
                )
            # logger.info(f"[Got LM prob] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            
            if args.loss_type == "kl_div":
                loss = calculate_KL_div_loss(input_logits=retriever_cossim, target_logits=lm_prob, temperature=[args.ret_temperature, args.lm_temperature])
            elif args.loss_type == "rag":
                loss = calculate_nll_loss(doc_scores=retriever_cossim, seq_probs=lm_prob)
            else:
                loss = calculate_cross_entropy_loss(input_logits=retriever_cossim, target_logits=lm_prob, temperature=[args.ret_temperature, args.lm_temperature])
            total_loss += loss.item()
            logger.info(f"[Got {args.loss_type} loss] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            ## Metric 2. Average answer probability
            # for each question, take idx of max retriever_cossim 
            # get its corresponding lm_prob
            # note. retriever_cossim and lm_prob are both [n_question,n_comb]
            retrievers_pick = torch.argmax(retriever_cossim,dim=1) # [n_question]

            # # %%
            # # ### debug
            for i in range(min(10,num_orig_question)):
                print(f"retriever_cossim: {retriever_cossim[i]}")
                print(f"retriever's pick: {retrievers_pick[i]}")
                print(f"lm_prob: {lm_prob[i]}")
                print(f"retrievers pick lm score: {lm_prob[i][retrievers_pick[i]]}")
                print(f"lm score each question max: {lm_prob[i][torch.argmax(lm_prob[i])]}")
                print(f"Retriever pick == LM pick? {retrievers_pick[i] == torch.argmax(lm_prob[i])}")
            # print(f"softmax retriever score: {F.softmax(retriever_cossim / args.ret_temperature,dim=1)}")
            
            # print(f"lm_prob.shape: {lm_prob.shape}")
            # debug_prompt_ans_lm_inputs = batch['prompt_ans_lm_inputs']['input_ids'].view(num_orig_question, -1, batch['prompt_ans_lm_inputs']['input_ids'].shape[-1])
            # debug_prompt_ans_lm_token_type_ids = batch['prompt_ans_lm_inputs']['token_type_ids'].view(num_orig_question, -1, batch['prompt_ans_lm_inputs']['token_type_ids'].shape[-1])
            # for i in range(num_orig_question):
            #     print(f"softmax retriever score: {F.softmax(retriever_cossim[i] / 0.1, dim=0)}")
            #     print(f"retriever_pick: {retrievers_pick[i]}")
            #     print(f"orig lm prob: {lm_prob[i]}")
            #     print(f"softmax lm prob: {F.softmax(lm_prob[i] / 0.1, dim=0)}")
            #     print(f"token_type_ids: {debug_prompt_ans_lm_token_type_ids[i]}")
            #     # decode each batch['prompt_ans_lm_inputs']
            #     batch_decode_result = lm_tokenizer.batch_decode(debug_prompt_ans_lm_inputs[i], skip_special_tokens=True)
            #     for j, decode_result in enumerate(batch_decode_result):
            #         print(f"{j}th decoded: {decode_result}")
            # print(f"lm_prob: {lm_prob}")
            # print(f"softmax lm score: {F.softmax(lm_prob / args.lm_temperature,dim=1)}")

            # # print(f"retrievers_pick: {retrievers_pick}")
            # print(f"retrievers_pick.shape: {retrievers_pick.shape}")
            # # decode each batch['prompt_ans_lm_inputs']
            # print(f"lm score each question max: {lm_prob[torch.arange(num_orig_question),torch.argmax(lm_prob,dim=1)]}")
            # print(f"retrievers pick lm score: {lm_prob[torch.arange(num_orig_question),retrievers_pick]}")
            # # ### debug
            # %%
            total_num_correct_pick += (retrievers_pick == torch.argmax(lm_prob,dim=1)).sum().item()
            lm_prob = lm_prob[torch.arange(num_orig_question),retrievers_pick] # [n_question]
            total_ans_prob += lm_prob.sum().item() 
            # count how many retriever's pick is the same as lm's pick
            all_retriever_pick.extend(retrievers_pick.tolist())
            retriever_cossim, lm_prob = retriever_cossim.to("cpu"), lm_prob.to("cpu")
            del retriever_cossim, lm_prob
            torch.cuda.empty_cache()
            gc.collect()
            # logger.info(f"[Emptied scoring cache] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

            # ## Metric 3. Exact match
            # reshape batch['prompt_ans_lm_inputs'] to [n_question,n_comb,n_dim]
            # %%
            # # debug: check if the picked one is really the same as the max one
            # for i in range(batch['prompt_ans_lm_inputs']['input_ids'].shape[0] // num_orig_question):
            #     tmp_str = lm_tokenizer.decode(batch['prompt_ans_lm_inputs']['input_ids'][i], skip_special_tokens=True)
            #     logger.debug(f"{i}th option: {tmp_str}")
            # logger.debug(f"Picked retrievers_pick idx: {retrievers_pick[0]}")

            logger.debug(f'batch["prompt_ans_lm_inputs"]["input_ids"].shape: {batch["prompt_ans_lm_inputs"]["input_ids"].shape}')
            n_comb = batch["prompt_ans_lm_inputs"]["input_ids"].shape[0] // num_orig_question
            logger.debug(f"n_comb: {n_comb}")
            batch['prompt_ans_lm_inputs'] = {
                k: v.view(num_orig_question, -1, v.shape[-1])[torch.arange(num_orig_question),retrievers_pick] \
                for k,v in batch['prompt_ans_lm_inputs'].items()
            } # [n_question,n_comb,n_dim] -> [n_question,n_dim]
            assert batch["prompt_ans_lm_inputs"]["input_ids"].shape[0] == num_orig_question, f"batch['prompt_ans_lm_inputs']['input_ids'].shape[0] ({batch['prompt_ans_lm_inputs']['input_ids'].shape[0]}) != num_orig_question ({num_orig_question})"
            # tmp_str = lm_tokenizer.decode(batch['prompt_ans_lm_inputs']['input_ids'][0], skip_special_tokens=True)
            # logger.debug(f"Picked option: {tmp_str}")

            # only leave the retriever's pick full_answers, all_qid, prompt_strs
            # use it to index full_answers, all_qid, prompt_strs
            if "llama" in args.lm_model.lower():
                batch["prompt_strs"] = [batch["prompt_strs"][pick + i * n_comb] for i, pick in enumerate(retrievers_pick)]
                assert len(batch["prompt_strs"]) == num_orig_question, f"len(batch['prompt_strs']) ({len(batch['prompt_strs'])}) != num_orig_question ({num_orig_question})"

            # %%
            # TODO (優化) llama 好像其實不用 prompt_ans_lm_inputs, 只要 prompt_strs 就好
            batch_result = lm_gen_and_check(
                model=language_model, 
                tokenizer=lm_tokenizer,
                device=accelerator.device,
                accelerator=accelerator,
                max_length=model_max_length,
                prompt_ans_lm_inputs=batch['prompt_ans_lm_inputs'], # for t5
                prompt_strs = batch["prompt_strs"] if "llama" in args.lm_model.lower() else None, # for llama
                all_full_answers=batch["full_answers"],
                max_tokens_to_generate=args.max_tokens_to_generate,
                llm_batch_size=args.eval_llm_batch_size,
                logger=logger,
            )
            total_num_correct += batch_result["num_correct"]
            total_num_examples += batch_result["num_examples"]
            total_too_long += batch_result["too_long"]
            all_predictions.extend(batch_result["predictions"])
            total_has_answer += batch_result["num_has_answer"]
            total_f1_score += batch_result["sum_f1"]

    # %%
    # write retriever pick to train_step_logdir
    with open(os.path.join(train_step_logdir, "retriever_pick.txt"), "w") as f:
        for pick in all_retriever_pick:
            f.write(str(pick) + "\n")
    with open(os.path.join(train_step_logdir, "prediction.json"), "w", encoding='utf-8') as f:
        for item in all_predictions:
            f.write(item + "\n")
    final_result = {
        "avg_loss": total_loss / len(dev_dataloader), # 這裡原本算錯啦! 應該以 batch 為單位才對
        "avg_prob": total_ans_prob / total_num_examples, # 這裡原本算錯啦! 原本是每個 batch 的 mean 加起來再除以 num_batches
        "exact_match (%)": total_num_correct / total_num_examples * 100,
        "too_long (%)": total_too_long / total_num_examples * 100,
        "has_answer (%)": total_has_answer / total_num_examples * 100, # 這裡原本算錯啦! 原本是所有 comb 的都算 但其實應該只能看選出來的那個
        "retriever_pick_acc (%)": total_num_correct_pick / total_num_examples * 100,
        "f1_score": total_f1_score / total_num_examples,
    }
    logger.info(f"Done {train_step_logdir.split('/')[-1]} step validation.")
    logger.info(f"total_num_examples: {total_num_examples}")
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
        mixed_precision='bf16', # turn on bf16
        kwargs_handlers=[kwargs]
    )
    logger.debug("*** IN DEBUG MODE ***")
    logger.info(f"device: {accelerator.device}")
    if "flan" in args.lm_model:
        model_short_name = "flan"
    elif "llama-3" in args.lm_model.lower():
        model_short_name = "llama3"
    elif "llama" in args.lm_model.lower():
        model_short_name = "llama"
    
    if args.resume_training:
        assert os.path.exists(args.resume_path), f"resume_path {args.resume_path} does not exist"
        logger.info(f"Resuming training from {args.resume_path}")
        # init tracker without config
        accelerator.init_trackers(
            project_name="dpr",
            init_kwargs={"wandb":{"id":args.resume_wandb_id, "resume":"must"}},
        )
    else:
        accelerator.init_trackers(
            project_name="dpr", 
            config=args,
            init_kwargs={"wandb":{"name":
                f"({args.data_size}) (lr {args.lr} warmup {args.warmup_steps}) ({args.empty_doc} empty) {model_short_name}-{args.max_round}round-{args.loss_type}-{args.k}k-bs({args.per_device_train_batch_size}&{args.per_device_eval_batch_size})({args.train_llm_batch_size}&{args.eval_llm_batch_size}) {args.max_train_epochs}ep doc({args.doc_encoder_type}) query({args.query_encoder_type})"}}
        )
    # %%
    if not debug and accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir
        if args.sweep:
            # exit if folder already exists
            CKPT_DIR = os.path.join(args.ckpt_dir, f"test-fit-lr-{args.lr}")
            if os.path.exists(CKPT_DIR):
                logger.info(f"CKPT_DIR {CKPT_DIR} already exists, exit successfully.")
                return
        else:
            CKPT_DIR = os.path.join(args.ckpt_dir, wandb_tracker.run.id)
        os.makedirs(CKPT_DIR)
        logger.info(f"Logging to {LOG_DIR}, create CKPT_DIR {CKPT_DIR}...")
        wandb_tracker.run.log_code(".")
        if not args.resume_training:
            wandb_tracker.run.tags = [
                f"size: {args.data_size}", f"lm: {args.lm_model}", f"loss: {args.loss_type}",
                f"query_enc: {args.query_encoder}", f"doc_enc: {args.doc_encoder}", 
                f"max_round: {args.max_round}", f"k: {args.k}", f"epoch: {args.max_train_epochs}", 
                f"train_bs: {args.per_device_train_batch_size}", f"eval_bs: {args.per_device_eval_batch_size}",
                f"temp: {args.ret_temperature}&{args.lm_temperature}","newline_format_prompt", "train", 
                f"empty_doc: {args.empty_doc}",
                "cossim_ret_score (correct)"
            ]
        else:
            # make sure current param is the same as the resumed one
            # except for resume_training, resume_path, resume_wandb_id
            exception_keys = ["resume_training", "resume_path", "resume_wandb_id"]
            for k,v in vars(args).items():
                logger.info(f"args.{k}: {v}")
                if k not in exception_keys:
                    assert wandb_tracker.run.config[k] == v, \
                    f"config {k} is different from resumed one: {wandb_tracker.run.config[k]} != {v}"
            assert args.resume_wandb_id in args.resume_path, f"resume_wandb_id not in resume_path: {args.resume_wandb_id} not in {args.resume_path}"
    else:
        LOG_DIR = "./tmp_log"  # Or any other directory you want to use when debugging
        CKPT_DIR = "./tmp_ckpt"
    
    # %%
    query_tokenizer, query_encoder = load_query_encoder_and_tokenizer(args, logger)

    if not debug and accelerator.is_local_main_process:
        wandb_tracker.run.watch(query_encoder, log_freq=500)
    # %%

    logger.info("...Loading language models...")
    language_model, lm_tokenizer, lm_config = load_lm_model_and_tokenizer(
        args.lm_model, device=accelerator.device, quantized=args.quantized,
        model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )
    language_model.eval()

    model_max_length = lm_config.n_positions if hasattr(lm_config, "n_positions") else lm_config.max_position_embeddings
    # only pad if model is gpt2
    if "gpt2" in args.lm_model or "llama" in args.lm_model:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
        # lm_tokenizer.pad_token = "[PAD]"
        lm_tokenizer.padding_side = "left" # TODO in llama1, should pad left??
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

    if args.data_size == "debug":
        train_size, dev_size = 50, 10
    elif args.data_size == "debug-fit-1":
        train_size, dev_size = 100, 100
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
    train_data = json.load(open(os.path.join(args.train_dir, args.train_file)))
    dev_data = json.load(open(os.path.join(args.dev_dir, args.dev_file)))
    logger.info(f"Size of train data: {len(train_data)}")
    logger.info(f"Size of dev data: {len(dev_data)}")

    logger.info("...Creating Corpus...")
    train_corpus = [[x['text'] for x in sample['ctxs']] for sample in train_data]
    dev_corpus = [[x['text'] for x in sample['ctxs']] for sample in dev_data]
    logger.info(f"Size of train corpus: {len(train_corpus)}")
    logger.info(f"Size of dev corpus: {len(dev_corpus)}")

    index_dir = os.path.join(args.base_index_dir, args.doc_encoder_type)
    train_index_path = os.path.join(index_dir, f"train_{train_size}.pt")
    dev_index_path = os.path.join(index_dir, f"dev_{dev_size}.pt")
    empty_doc_embedding_path = os.path.join(index_dir, "empty_doc.pt")

    if os.path.exists(train_index_path) and os.path.exists(dev_index_path) and os.path.exists(empty_doc_embedding_path):
        logger.info(f"...Loading index from {train_index_path} and {dev_index_path}...") 
        # skip those exemplars
        train_doc_embeddings = torch.load(train_index_path)
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
            if not os.path.exists(train_index_path):
                logger.info(f"...Creating train index with size {len(train_corpus)}...")
                train_doc_embeddings = [make_index(corpus, doc_tokenizer, doc_encoder) for corpus in tqdm(train_corpus)]
                torch.save(train_doc_embeddings, train_index_path)
            else:
                logger.info(f"...Loading train index from {train_index_path}...")
                train_doc_embeddings = torch.load(train_index_path)
            if not os.path.exists(dev_index_path):
                logger.info(f"...Creating dev index with size {len(dev_corpus)}...")
                dev_doc_embeddings = [make_index(corpus, doc_tokenizer, doc_encoder) for corpus in tqdm(dev_corpus)]
                torch.save(dev_doc_embeddings, dev_index_path)
            else:
                logger.info(f"...Loading dev index from {dev_index_path}...")
                dev_doc_embeddings = torch.load(dev_index_path)
            if not os.path.exists(empty_doc_embedding_path):
                logger.info(f"...Creating empty embedding ...")
                empty_doc_embedding = make_index(["[UNK]"], doc_tokenizer, doc_encoder).squeeze() # for empty document
                torch.save(empty_doc_embedding, empty_doc_embedding_path)
            else:
                logger.info(f"...Loading empty embedding ...")
                empty_doc_embedding = torch.load(empty_doc_embedding_path)
        logger.info(f"Index saved to {train_index_path}, {dev_index_path}, {empty_doc_embedding_path}")
        logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

        logger.info("...Deleting doc_encoder...")
        doc_encoder = doc_encoder.to("cpu")
        del doc_encoder
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

    # take the [args.num_exemplars:] 
    train_data = train_data[args.num_exemplars:]
    dev_data = dev_data[args.num_exemplars:]
    train_corpus = train_corpus[args.num_exemplars:]
    dev_corpus = dev_corpus[args.num_exemplars:]
    train_doc_embeddings = train_doc_embeddings[args.num_exemplars:]
    dev_doc_embeddings = dev_doc_embeddings[args.num_exemplars:]

    # # %%
    # gold_path = os.path.join(LOG_DIR, args.gold_dev_answers_path)
    # if not os.path.exists(gold_path):
    #     logger.info(f"...Creating gold answers for dev set...")
    #     ensure_directory_exists_for_file(gold_path)
    #     gold_answers = []
    #     for sample in dev_data:
    #         gold_answers.append(sample['answers']) # log all answer 
    #         # gold_answers.append(sample['answers'][0])
    #     with open(gold_path, "w") as f:
    #         for ans in gold_answers:
    #             f.write(str(ans) + "\n")
    #     logger.info(f"Gold answers saved to {gold_path}")
    #     del gold_answers

    # TODO add feature of empty doc representation

    # Answer is a LIST instead of a str
    # In train set, only select one answer for each question
    train_qa_pairs = [(normalize_query(sample['question']), [""], [sample['answers'][0]], empty_doc_embedding) for sample in train_data]
    # In dev set, keep track of full answer list of each question
    dev_qa_pairs = [(normalize_query(sample['question']), [""], sample['answers'], empty_doc_embedding) for sample in dev_data]

    logger.info(f"len(train_qa_pairs): {len(train_qa_pairs)}")
    logger.info(f"len(dev_qa_pairs): {len(dev_qa_pairs)}")
    logger.info(f"len(train_corpus): {len(train_corpus)}")
    logger.info(f"len(dev_corpus): {len(dev_corpus)}")
    logger.info(f"len(train_doc_embeddings): {len(train_doc_embeddings)}")
    logger.info(f"len(dev_doc_embeddings): {len(dev_doc_embeddings)}")

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

    # %%
    if args.resume_training:
        logger.info(f"...Loading old state_dict from ckpt {args.resume_path}...")
        state_dict = torch.load(args.resume_path)
        query_encoder.load_state_dict(state_dict["query_encoder"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        completed_steps = state_dict["completed_steps"]
        logger.info(f"...State_dict at step {completed_steps} loaded to query_encoder, optimizer, lr_scheduler...")
    else:
        logger.info(f"\n...0 Step Evaluation...")
        train_step_logdir = os.path.join(LOG_DIR,f"step-{completed_steps}")
        if not os.path.exists(train_step_logdir):
            os.makedirs(train_step_logdir)
    # %%
        eval_result = validate(query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, train_step_logdir)
        accelerator.log({"eval":eval_result}, step=completed_steps)
    best_em = eval_result["exact_match (%)"]

    # %%
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
    logger.info(f"  Train LM batch size = {args.train_llm_batch_size}")
    logger.info(f"  Eval LM batch size = {args.eval_llm_batch_size}")
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    start_time = time.time()

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        logger.info(f"[Before load train data] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        
        for step,raw_batch in enumerate(train_dataloader):
            # make raw_batch into a extened batch
            # by first extend each item and then collate_fn
            extended_batch = [inloop_extend_item(
                data=x["data"], corpus=x["corpus"], doc_embeddings=x["doc_embeddings"],
                ret_tokenizer=query_tokenizer, query_encoder=query_encoder, args=args
            ) for x in raw_batch]
            batch = inloop_collate_fn(
                samples=extended_batch, ret_tokenizer=query_tokenizer, lm_tokenizer=lm_tokenizer, 
                lm_name=args.lm_model, args=args, mode="train"
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
                with accelerator.autocast(): # mixed precision
                    # logger.debug(f"batch['query_inputs']['input_ids']: {batch['query_inputs']['input_ids'].shape}")
                    # logger.debug(f"batch['doc_embeddings']: {batch['doc_embeddings'].shape}")
                    query_embedding = query_encoder(**batch['query_inputs']).pooler_output \
                        if "dpr" in args.query_encoder \
                        else query_encoder(**batch['query_inputs']).last_hidden_state[:,0,:]
                    doc_embedding = batch["doc_embeddings"]
                    # logger.info(f"[Sent to query encoder] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                    
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
                    num_orig_question = single_device_query_num // sum([args.k ** i for i in range(args.max_round + 1)]) if args.empty_doc \
                        else single_device_query_num // (sum([args.k ** i for i in range(args.max_round + 1)]) - 1)
                    retriever_cossim = retriever_cossim.reshape(num_orig_question, -1)
                    # logger.info(f"[Got ret cos sim] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")

                    query_embedding, doc_embedding = query_embedding.to("cpu"), doc_embedding.to("cpu")
                    del query_embedding, doc_embedding
                    torch.cuda.empty_cache()
                    gc.collect()
                    # logger.info(f"[Emptied embedding cache] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

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
                    # logger.info(f"[Got LM prob] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
                    
                    # # try to fix RuntimeError: Found dtype Float but expected Half
                    # for param in query_encoder.parameters():
                    #     # Check if parameter dtype is  Float (float32)
                    #     if param.dtype == torch.float32:
                    #         param.data = param.data.to(torch.float16)
                    # try to fix RuntimeError: Found dtype Float but expected Half
                    # retriever_cossim = retriever_cossim
                    # lm_prob = lm_prob

                    if args.loss_type == "kl_div":
                        loss = calculate_KL_div_loss(input_logits=retriever_cossim, target_logits=lm_prob, temperature=[args.ret_temperature, args.lm_temperature])
                    elif args.loss_type == "rag":
                        loss = calculate_nll_loss(doc_scores=retriever_cossim, seq_probs=lm_prob)
                    else:
                        loss = calculate_cross_entropy_loss(input_logits=retriever_cossim, target_logits=lm_prob, temperature=[args.ret_temperature, args.lm_temperature])
                    logger.info(f"[Got {args.loss_type} loss] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")

                    retriever_cossim, lm_prob = retriever_cossim.to("cpu"), lm_prob.to("cpu")
                    del retriever_cossim, lm_prob
                    torch.cuda.empty_cache()
                    gc.collect()

                accelerator.backward(loss)
                logger.info(f"[After backward] loss = {loss}; GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB. Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")

                # one optimization step
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
                        # logger.info(f"[Before evaluation] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB.  Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
                        logger.info(f"[Before evaluation] max ret token len: {max_ret_token_len}; max lm token len: {max_lm_token_len}")
                        train_step_logdir = os.path.join(LOG_DIR,f"step-{completed_steps}")
                        if not os.path.exists(train_step_logdir):
                            os.makedirs(train_step_logdir)
                        eval_result = validate(query_tokenizer, query_encoder, language_model, dev_dataloader, lm_tokenizer, args, accelerator, model_max_length, train_step_logdir)
                        query_encoder.train() # Make sure the model is back in training mode after validation
                        accelerator.log({"eval":eval_result}, step=completed_steps)
                        accelerator.wait_for_everyone()
                        logger.info(f"[Got every eval subproc] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                        if accelerator.is_local_main_process:
                            # only save the best model to dist (don't save to wandb dir)
                            logger.info(f"best_em: {best_em}")
                            logger.info(f"exact_match: {eval_result['exact_match (%)']}")
                            if eval_result["exact_match (%)"] > best_em:
                                best_em = eval_result["exact_match (%)"]

                            ckpt_path = os.path.join(CKPT_DIR, f"checkpoint-{completed_steps}.pt")
                            ensure_directory_exists_for_file(ckpt_path)
                            # unwrap the model from DDP
                            unwrapped_model = accelerator.unwrap_model(query_encoder)
                            unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                            torch.save({
                                'query_encoder': unwrapped_model.state_dict(),
                                'optimizer': unwrapped_optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'completed_steps': completed_steps,}, ckpt_path)
                            logger.info(f"Checkpoint saved to {ckpt_path}")
                            
                        accelerator.wait_for_everyone()
                
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"[Finish step {step} in epoch {epoch} (globally {completed_steps})] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB.  Current Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
    
    if accelerator.is_local_main_process:
        logger.info(f"max_ret_token_len: {max_ret_token_len}; max_lm_token_len: {max_lm_token_len}")
        logger.info(f"Time spent: {time.time() - start_time} seconds")
        logger.info(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e6} MB")
        logger.info("...!!Congrats!! Training finished :) ...")
        logger.info(f"Checkpoint saved to {CKPT_DIR}")
        if not debug:
            wandb_tracker.finish()
    
    accelerator.end_training()

# %%
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn') # try to fix the cuda init error when we put query encoder on cuda
    main()