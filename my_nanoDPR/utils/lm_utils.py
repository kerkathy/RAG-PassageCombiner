import json
import os
import re
import string
import torch
from transformers import AutoConfig, AutoTokenizer

def load_lm_tokenizer(model_name):
    if "llama" in model_name:
        from transformers import LlamaTokenizer
        return LlamaTokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)

def load_lm_model_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CPU not supported!!!!"
    device_count = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    if "flan" in model_name:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_args).eval()
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)
    tokenizer = load_lm_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, config, device

def get_lm_score(
    model, 
    device,
    input_ids,
    attention_mask,
    token_type_ids,
    max_length, max_tokens_to_generate, 
    num_orig_question,
    llm_batch_size=1,
):
    """
    Take a model and a batch of input_ids, attention_mask, and token_type_ids and return the log probability of the input_ids
    The return value is NOT a distribution, but a log probability
    Note: ext_batch_size here = num_orig_question * n_comb
    """
    if input_ids.shape[-1] > max_length - max_tokens_to_generate:
        num_too_long += 1
        input_ids = input_ids[..., -(max_length - max_tokens_to_generate):]

    all_outputs = []
    for input_ids_batch, attention_mask_batch, token_type_ids_batch in zip(input_ids.split(llm_batch_size), attention_mask.split(llm_batch_size), token_type_ids.split(llm_batch_size)):
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        token_type_ids_batch = token_type_ids_batch.to(device)
        with torch.no_grad():
            if "flan" in model.name_or_path:
                outputs_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, decoder_input_ids=input_ids_batch).logits
            else:
                outputs_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
            outputs_batch = torch.log_softmax(outputs_batch, dim=-1).detach()
        
        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        outputs_batch, input_ids_batch = outputs_batch[:, :-1, :], input_ids_batch[:, 1:]
        outputs_batch = torch.gather(outputs_batch, 2, input_ids_batch[:, :, None]).squeeze(-1) # [ext_batch_size, seq_len, 1] -> [ext_batch_size, seq_len]
        all_outputs.append(outputs_batch)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    token_type_ids = token_type_ids[:, 1:]
    
    # set the log probs to 0 where token_type_ids = 0
    all_outputs[token_type_ids == 0] = 0 # [ext_batch_size, seq_len]

    # compute sequence scores
    # option 1. sum of neg log probabilities
    all_outputs = all_outputs.sum(dim=-1).view(num_orig_question, -1) # TODO: exp or not
    # option 2. joint probability
    # joint_probs = probs.sum(dim=-1).view(num_orig_question, -1).exp() # [num_orig_question, n_comb]

    return all_outputs # [num_orig_question, n_comb]

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_batch_answer_from_model_output(outputs, tokenizer, prompt_lengths):
    # Use batch_decode to decode all outputs at once
    generation_strs = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
    
    answers = []
    for i, generation_str in enumerate(generation_strs):
        generation_str = generation_str[prompt_lengths[i]:]
        answer = generation_str.split("\n")[0]
        answers.append(answer)
    return answers

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_dataset(
        model, tokenizer, device, max_length, prompt_ans_lm_inputs,
        max_tokens_to_generate=10, steps_log_dir=".", llm_batch_size=1
):
    num_correct = 0
    num_too_long = 0
    all_predictions = []

    input_ids = prompt_ans_lm_inputs["input_ids"].to(device)
    token_type_ids = prompt_ans_lm_inputs["token_type_ids"].to(device) # token_type_ids = 0 for prompt, 1 for answer

    # get decoded answers
    answers = [tokenizer.decode(input_ids[i, token_type_ids[i] == 1], skip_special_tokens=True) for i in range(input_ids.shape[0])]

    # extract prompt ids and prompt lengths from input_ids
    range_tensor = torch.arange(input_ids.size(1)).expand_as(input_ids).to(device)
    ans_start_pos = (token_type_ids == 1).float().argmax(dim=1) # [bs]
    prompt_mask = range_tensor < ans_start_pos.unsqueeze(1) # broadcast to [bs, seq_len]
    all_prompt_input_ids = input_ids * prompt_mask # [bs, seq_len]
    all_prompt_input_ids[all_prompt_input_ids == 0] = tokenizer.pad_token_id
    prompt_lengths = prompt_mask.sum(dim=1)
    del prompt_mask, range_tensor, ans_start_pos, input_ids, token_type_ids

    for prompt_input_ids in all_prompt_input_ids:
        if prompt_input_ids.shape[-1] > max_length - max_tokens_to_generate:
            num_too_long += 1
            prompt_input_ids = prompt_input_ids[..., -(max_length - max_tokens_to_generate):]

    for input_ids_batch, prompt_length_batch in zip(all_prompt_input_ids.split(llm_batch_size), prompt_lengths.split(llm_batch_size)):
        with torch.no_grad():
            output_batch = model.generate(input_ids_batch, max_new_tokens=max_tokens_to_generate)
            all_predictions.extend(get_batch_answer_from_model_output(output_batch, tokenizer, prompt_length_batch))

    for prediction, answer in zip(all_predictions, answers):
        is_correct = exact_match(prediction, answer) # now we only have one ans per example
        # is_correct = any([exact_match(prediction, answer) for answer in answers])
        if is_correct:
            num_correct += 1

    num_data = len(prompt_lengths)
    em = num_correct / num_data * 100
    # print(f"EM: {em:.1f}%")

    d = {"em": em, "num_examples": num_data, "too_long": num_too_long}

    with open(os.path.join(steps_log_dir, "eval.json"), "w", encoding='utf-8') as f:
        f.write(json.dumps(d) + "\n")
    with open(os.path.join(steps_log_dir, "prediction.json"), "w", encoding='utf-8') as f:
        for item in all_predictions:
            f.write(item + "\n")

    return d