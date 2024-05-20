import re
import string
import torch
from torch.nn import CrossEntropyLoss

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

# %%
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + '▶')
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
# %%

def exact_match(prediction, ground_truth):
    # TODO 考慮改寬鬆一點
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False

# %%
def separate_prompt_answer(input_ids, token_type_ids, tokenizer, device):
    """
    Consider input_ids and token_type_ids as a batch of prompt-answer pairs.
    Given input_ids and token_type_ids, return the prompt input_ids and prompt lengths.
    Optionally return the answer input_ids as well.
    """
    input_ids[token_type_ids == 1] = tokenizer.pad_token_id

    # old version
    # ans_start_pos = (token_type_ids == 1).float().argmax(dim=1) # [bs]
    # prompt_mask = torch.arange(input_ids.size(1)).expand_as(input_ids).to(device) < ans_start_pos.unsqueeze(1).to(device) # broadcast to [bs, seq_len]
    # mask the answer part into pad tokens
    # prompt_input_ids = input_ids * prompt_mask.to(device) # [bs, seq_len] #
    # prompt_input_ids[prompt_input_ids == 0] = tokenizer.pad_token_id
    # Count only the prompt token length (excluding the answer part and the padding tokens)
    # prompt_ids_lengths = prompt_input_ids.ne(tokenizer.pad_token_id).sum(dim=1) # [bs]
    # create attention mask where 0 is for padding tokens
    # attention_mask = prompt_input_ids.ne(tokenizer.pad_token_id).float() # [bs, seq_len]

    return input_ids

# %%

def get_t5_lm_prob(
    input_ids, labels,
    model, device, tokenizer,
    max_length, max_tokens_to_generate, 
    num_orig_question, llm_batch_size=1, logger=None
):
    """
    Take a model and a batch of input_ids, attention_mask, and token_type_ids,
    return the log probability of the input_ids.

    The return value is NOT a distribution, but a log probability
    Note: ext_batch_size here = num_orig_question * n_comb

    This function is for T5 models (T5, flan-T5, ...) only!
    As T5 is an encoder-decoder model, we need to provide input_ids to the encoder and decoder separately.
    Here, we assume the input_ids are prompt-answer pairs, and we need to separate them into prompt and answer.
    Then we feed the prompt to the encoder and the answer to the decoder.
    In that case, output of decoder is of has length of the answer, rather than the prompt_ans pair.
    """
    # print("...In get_t5_lm_prob...")
    # print("input_ids.shape: ", input_ids.shape)
    # print("labels.shape: ", labels.shape)
    # print("llm_batch_size: ", llm_batch_size)

    num_too_long = 0
    # print(f"Max accepted length: {max_length} - {max_tokens_to_generate} = {max_length - max_tokens_to_generate}")
    if labels.shape[-1] > max_length - max_tokens_to_generate:
        num_too_long += 1
        labels = labels[..., -(max_length - max_tokens_to_generate):]
        print("labels.shape: ", labels.shape)
    if num_too_long > 0:
        print("Num too long: ", num_too_long)

    log_probs = []
    for input_ids_batch, labels_batch \
        in zip(input_ids.split(llm_batch_size), labels.split(llm_batch_size)):

        input_ids_batch = input_ids_batch.to(device)
        labels_batch = labels_batch.to(device)
        with torch.no_grad():
            # t5 shifts labels to the right by one internally
            logits_batch = model(input_ids=input_ids_batch, labels=labels_batch).logits
            logger.info(f"[Passed LM] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            eff_batch_size, seq_len, vocab_size = logits_batch.shape
            ce_fn = CrossEntropyLoss(
                reduction="none", ignore_index=tokenizer.pad_token_id)
            log_probs_batch = -ce_fn(logits_batch.view(-1, vocab_size), labels_batch.view(-1)) # [eff_batch_size * seq_len]
            logger.info(f"[Loss calced] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            
        log_probs.append(log_probs_batch.view(eff_batch_size, seq_len).sum(dim=-1)) # [eff_batch_size]
        # print("T5 log_probs_batch.shape: ", log_probs[-1].shape)
    log_probs = torch.cat(log_probs, dim=0)
    # print("Concated log_probs.shape: ", log_probs.shape, "\n")
    # print("log_probs: ", log_probs)
    return log_probs.view(num_orig_question, -1).exp() # [num_orig_question, n_comb]""
    # return log_probs.view(num_orig_question, -1) # [num_orig_question, n_comb]""


def get_lm_prob(
    input_ids, attention_mask, token_type_ids,
    model, device, 
    max_length, max_tokens_to_generate, 
    num_orig_question, llm_batch_size=1, logger=None
):
    """
    Take a model and a batch of input_ids, attention_mask, and token_type_ids,
    return the log probability of the input_ids.
    The return value is NOT a distribution, but a log probability.
    Note: ext_batch_size here = num_orig_question * n_comb
    """
    num_too_long = 0
    if input_ids.shape[-1] > max_length - max_tokens_to_generate:
        num_too_long += 1
        input_ids = input_ids[..., -(max_length - max_tokens_to_generate):]
        attention_mask = attention_mask[..., -(max_length - max_tokens_to_generate):]
        token_type_ids = token_type_ids[..., -(max_length - max_tokens_to_generate):]
    if num_too_long > 0:
        print("Num too long: ", num_too_long)

    all_outputs = []
    for input_ids_batch, attention_mask_batch \
        in zip(input_ids.split(llm_batch_size), attention_mask.split(llm_batch_size)):

        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)

        with torch.no_grad():
            outputs_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits # [ext_batch_size, seq_len, vocab_size]
            logger.info(f"[Passed LM] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            outputs_batch = torch.log_softmax(outputs_batch, dim=-1).detach()
        
        # collect the probability of the generated token
        # (probability at index 0 corresponds to the token at index 1)
        outputs_batch, input_ids_batch = outputs_batch[:, :-1, :], input_ids_batch[:, 1:]
        outputs_batch = torch.gather(outputs_batch, 2, input_ids_batch[:, :, None]).squeeze(-1) # [ext_batch_size, seq_len, 1] -> [ext_batch_size, seq_len]
        all_outputs.append(outputs_batch)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    token_type_ids = token_type_ids[:, 1:]
    
    # set the log probs to 0 where token_type_ids = 0
    all_outputs[token_type_ids == 0] = 0 # [ext_batch_size, seq_len]

    # compute sequence scores
    # option 1. sum of log probabilities
    # all_outputs = all_outputs.sum(dim=-1).view(num_orig_question, -1) # exp or not...?
    # option 2. joint probability
    all_outputs = all_outputs.sum(dim=-1).view(num_orig_question, -1).exp() # [num_orig_question, n_comb]

    return all_outputs # [num_orig_question, n_comb]

# %%
def get_batch_answer_from_model_output(generation_strs, prompt_lengths):
    """
    For evaluation. Given a batch of model outputs, return the answer strings
    """
    answers = []
    for i, generation_str in enumerate(generation_strs):
        # print(f"{i}th uncut generation_str: {generation_str}")
        # fix!!! 好像有差一點 10 token (max_gen_token?)
        # fix!! 因為 pad 左側所以會有問題
        generation_str = generation_str[prompt_lengths[i]:]
        answer = generation_str.split("\n")[0]
        answers.append(answer)
        print(f"{i}th cutted generation_str: {generation_str}; answer: {answer}")
    return answers
# %%


def lm_gen_and_check(
        model, tokenizer, device, max_length, prompt_ans_lm_inputs,
        accelerator, max_tokens_to_generate=10, llm_batch_size=1, logger=None
):

    # %%
    num_too_long = 0
    all_predictions = []

    if "llama" in tokenizer.name_or_path:
        input_ids = prompt_ans_lm_inputs["input_ids"].to(device)
        token_type_ids = prompt_ans_lm_inputs["token_type_ids"].to(device) # token_type_ids = 0 for prompt, 1 for answer
        all_mask = prompt_ans_lm_inputs["attention_mask"].to(device)
        answers = [tokenizer.decode(input_ids[i, token_type_ids[i] == 1], skip_special_tokens=True) for i in range(input_ids.shape[0])]
        
        all_prompt_input_ids = input_ids
        all_prompt_input_ids[token_type_ids == 1] = tokenizer.pad_token_id # new
        all_mask[token_type_ids == 1] = 0 # new
        # all_prompt_input_ids = separate_prompt_answer(input_ids, token_type_ids, tokenizer, device) # old
        # %%
        token_type_ids = token_type_ids.to("cpu")
        del prompt_ans_lm_inputs, token_type_ids
        torch.cuda.empty_cache()

        # %%
        # constraint the length of the prompt
        for i, prompt_input_ids in enumerate(all_prompt_input_ids):
            if prompt_input_ids.shape[-1] > max_length - max_tokens_to_generate:
                num_too_long += 1
                all_prompt_input_ids[i] = prompt_input_ids[..., -(max_length - max_tokens_to_generate):]
                all_mask[i] = all_mask[i][..., -(max_length - max_tokens_to_generate):]

        # %%
        # new version: decode the prompt to get original prompt
        # and then encode it one by one, to avoid the padding issue
        for i, (prompt_input_ids, mask) in enumerate(zip(all_prompt_input_ids, all_mask)):
            prompt_str = tokenizer.decode(prompt_input_ids, skip_special_tokens=True)
            # tokenize
            prompt_input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                if "llama-3" in tokenizer.name_or_path.lower():
                    output = model.generate(
                        prompt_input_ids, 
                        max_new_tokens=max_tokens_to_generate, 
                        pad_token_id=tokenizer.eos_token_id, 
                        eos_token_id=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        attention_mask=mask.unsqueeze(0)
                    )

                else:
                    output = model.generate(prompt_input_ids, max_new_tokens=max_tokens_to_generate)
            generation_str = tokenizer.decode(output[0], skip_special_tokens=True)
            answer = generation_str.replace(prompt_str, "").split("\n")[0]
            all_predictions.append(answer)
            # logger.debug(f"prompt_str: {prompt_str}")
            # logger.debug(f"generation_str: {generation_str}")
            # logger.debug(f"answer: {answer}")

        # # below: old version 
        # for prompt_input_ids_batch in all_prompt_input_ids.split(llm_batch_size):
        #     with torch.no_grad():
        #         # add support to accelerator unwrap
        #         if hasattr(model, "module"):
        #             output_batch = accelerator.unwrap_model(model).generate(prompt_input_ids_batch, max_new_tokens=max_tokens_to_generate)
        #             logger.info(f"[Passed DDP LM] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        #         else:
        #             output_batch = model.generate(prompt_input_ids_batch, max_new_tokens=max_tokens_to_generate)
        #             logger.info(f"[Passed normal LM] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
        #     generation_strs = tokenizer.batch_decode(output_batch.cpu(), skip_special_tokens=True)
        #     prompt_strs = tokenizer.batch_decode(prompt_input_ids_batch.cpu(), skip_special_tokens=True)
        #     # remove the prompt part from the generated output because the model repeats the prompt
        #     for i, (generation_str, prompt_str) in enumerate(zip(generation_strs, prompt_strs)):
        #         answer = generation_str.replace(prompt_str, "").split("\n")[0]
        #         all_predictions.append(answer)
        #         logger.debug(f"generation_str: {generation_str}\nanswer: {answer}")

        #     all_prompt_strs.extend(prompt_strs)
    # %%
    else:
        # t5 generation has no constraint on the length
        answers = tokenizer.batch_decode(prompt_ans_lm_inputs["labels"], skip_special_tokens=True)
        # all_prompt_lengths = (prompt_ans_lm_inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
        all_prompt_input_ids = prompt_ans_lm_inputs["input_ids"].to(device)
        for input_ids_batch in all_prompt_input_ids.split(llm_batch_size):
            with torch.no_grad():
                # add support to accelerator unwrap
                if hasattr(model, "module"):
                    output_batch = accelerator.unwrap_model(model).generate(input_ids_batch, max_new_tokens=max_tokens_to_generate)
                    logger.info(f"[Passed DDP LM] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
                else:
                    output_batch = model.generate(input_ids_batch, max_new_tokens=max_tokens_to_generate)
                    logger.info(f"[Passed normal LM] GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
            generation_strs = tokenizer.batch_decode(output_batch.cpu(), skip_special_tokens=True)
            all_predictions.extend(generation_strs)


    # %%
    num_correct = 0
    for prediction, answer in zip(all_predictions, answers):
        is_correct = exact_match(prediction, answer) # now we only have one ans per example
        # is_correct = any([exact_match(prediction, answer) for answer in answers])
        if is_correct:
            num_correct += 1

    result = {"num_correct": num_correct, "num_examples": len(answers), "too_long": num_too_long, "predictions": [normalize_answer(prediction) for prediction in all_predictions]}
    # %%
    return result