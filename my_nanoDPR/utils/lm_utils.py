import re
import string
import torch
from torch.nn import CrossEntropyLoss

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

def separate_prompt_answer(input_ids, token_type_ids, tokenizer, device, return_ans=False):
    """
    Consider input_ids and token_type_ids as a batch of prompt-answer pairs.
    Given input_ids and token_type_ids, return the prompt input_ids and prompt lengths.
    Optionally return the answer input_ids as well.
    """
    ans_start_pos = (token_type_ids == 1).float().argmax(dim=1) # [bs]
    prompt_mask = torch.arange(input_ids.size(1)).expand_as(input_ids).to(device) < ans_start_pos.unsqueeze(1).to(device) # broadcast to [bs, seq_len]
    prompt_input_ids = input_ids * prompt_mask.to(device) # [bs, seq_len]
    prompt_input_ids[prompt_input_ids == 0] = tokenizer.pad_token_id
    prompt_lengths = prompt_mask.sum(dim=1)
    # decode the prompt ids to check if the prompt is correct
    prompt_strs = tokenizer.batch_decode(prompt_input_ids.cpu(), skip_special_tokens=True)
    # print("Separated prompt: ", prompt_strs)

    if return_ans:
        ans_input_ids = input_ids * ~prompt_mask.to(device) # [bs, seq_len]
        # decode the answer ids to check if the answer is correct
        ans_strs = tokenizer.batch_decode(ans_input_ids.cpu(), skip_special_tokens=True)
        # print("Separated answer: ", ans_strs)
        return prompt_input_ids, prompt_lengths, ans_input_ids

    return prompt_input_ids, prompt_lengths

def get_t5_lm_score(
    input_ids, labels,
    model, device, tokenizer,
    max_length, max_tokens_to_generate, 
    num_orig_question, llm_batch_size=1,
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
    # print("...In get_t5_lm_score...")
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
            logits_batch = model(input_ids=input_ids_batch, labels=labels_batch).logits
            eff_batch_size, seq_len = logits_batch.shape[:2]
            ce_fn = CrossEntropyLoss(
                reduction="none", ignore_index=tokenizer.pad_token_id)
            log_probs_batch = -ce_fn(
                logits_batch.view(-1, logits_batch.shape[-1]), labels_batch.view(-1))
            
        log_probs.append(log_probs_batch.view(eff_batch_size, -1, seq_len).sum(dim=-1)) # [n_batch_question, n_comb]
        # print("log_probs_batch.shape: ", log_probs[-1].shape)
    log_probs = torch.cat(log_probs, dim=0)
    # print("Concated log_probs.shape: ", log_probs.shape, "\n")
    # print("log_probs: ", log_probs)
    return log_probs.view(num_orig_question, -1) # [num_orig_question, n_comb]""


def get_lm_score(
    input_ids, attention_mask, token_type_ids,
    model, device, 
    max_length, max_tokens_to_generate, 
    num_orig_question, llm_batch_size=1,
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
    if num_too_long > 0:
        print("Num too long: ", num_too_long)

    all_outputs = []
    for input_ids_batch, attention_mask_batch in zip(input_ids.split(llm_batch_size), attention_mask.split(llm_batch_size)):
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)

        with torch.no_grad():
            outputs_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits # [ext_batch_size, seq_len, vocab_size]
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
    all_outputs = all_outputs.sum(dim=-1).view(num_orig_question, -1) # TODO: exp or not
    # option 2. joint probability
    # joint_probs = probs.sum(dim=-1).view(num_orig_question, -1).exp() # [num_orig_question, n_comb]

    return all_outputs # [num_orig_question, n_comb]


def get_batch_answer_from_model_output(generation_strs, prompt_lengths):
    """
    For evaluation. Given a batch of model outputs, return the answer strings
    """
    answers = []
    for i, generation_str in enumerate(generation_strs):
        generation_str = generation_str[prompt_lengths[i]:]
        answer = generation_str.split("\n")[0]
        answers.append(answer)
    return answers


def lm_gen_and_check(
        model, tokenizer, device, max_length, prompt_ans_lm_inputs,
        max_tokens_to_generate=10, train_step_logdir=".", llm_batch_size=1
):
    num_correct = 0
    num_too_long = 0
    all_predictions = []

    if "t5" in tokenizer.name_or_path:
        # t5 generation has no constraint on the length
        answers = tokenizer.batch_decode(prompt_ans_lm_inputs["labels"], skip_special_tokens=True)
        prompt_lengths = (prompt_ans_lm_inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
        all_prompt_input_ids = prompt_ans_lm_inputs["input_ids"].to(device)
    else:
        input_ids = prompt_ans_lm_inputs["input_ids"].to(device)
        token_type_ids = prompt_ans_lm_inputs["token_type_ids"].to(device) # token_type_ids = 0 for prompt, 1 for answer
        # get decoded answers
        answers = [tokenizer.decode(input_ids[i, token_type_ids[i] == 1], skip_special_tokens=True) for i in range(input_ids.shape[0])]
        # extract prompt ids and prompt lengths from input_ids
        all_prompt_input_ids, prompt_lengths = separate_prompt_answer(input_ids, token_type_ids, tokenizer, device, return_ans=False)
        del prompt_ans_lm_inputs
        # constraint the length of the prompt
        for prompt_input_ids in all_prompt_input_ids:
            if prompt_input_ids.shape[-1] > max_length - max_tokens_to_generate:
                num_too_long += 1
                prompt_input_ids = prompt_input_ids[..., -(max_length - max_tokens_to_generate):]

    if "llama" in tokenizer.name_or_path:
        # llama generation repeats the prompt, hence special handling
        for input_ids_batch, prompt_length_batch in zip(all_prompt_input_ids.split(llm_batch_size), prompt_lengths.split(llm_batch_size)):
            with torch.no_grad():
                output_batch = model.generate(input_ids_batch, max_new_tokens=max_tokens_to_generate)
            generation_strs = tokenizer.batch_decode(output_batch.cpu(), skip_special_tokens=True)
            all_predictions.extend(get_batch_answer_from_model_output(generation_strs, prompt_length_batch))
    else:
        for input_ids_batch in all_prompt_input_ids.split(llm_batch_size):
            with torch.no_grad():
                output_batch = model.generate(input_ids_batch, max_new_tokens=max_tokens_to_generate)
            generation_strs = tokenizer.batch_decode(output_batch.cpu(), skip_special_tokens=True)
            all_predictions.extend(generation_strs)

    for prediction, answer in zip(all_predictions, answers):
        # print("Prediction: ", prediction)
        # print("Answer: ", answer, "\n")
        is_correct = exact_match(prediction, answer) # now we only have one ans per example
        # is_correct = any([exact_match(prediction, answer) for answer in answers])
        if is_correct:
            num_correct += 1

    num_data = len(prompt_lengths)
    # em = num_correct / num_data * 100

    result = {"num_correct": num_correct, "num_examples": num_data, "too_long": num_too_long, "predictions": all_predictions}

    return result


def build_qa_prompt(example, num_docs=1, require_long=False, output_true_false=False):
    if output_true_false:
        # for strategyQA, we need to output true/false
        # don't care about num of doc
        q = normalize_question(example["question"])
        docs_text = "\n\n".join([ctx['text'] for ctx in example["ctxs"][:num_docs]])
        ex_prompt = f"""Given a question and a context, provide a Yes or No answer and explain why. If you are unsure, answer Unknown.
#
Context:
{docs_text}

Question:
{q}

Answer (Yes/No/Unknown):
"""
        
    elif num_docs == 0:
        question_text = normalize_question(example["question"])
        ex_prompt = f"Answer these questions:\nQ: {question_text}\nA:"
    elif num_docs == 1:
        q = normalize_question(example["question"])
        title = example['ctxs'][0]['title']
        if title == None:
            title = ""
        text = example['ctxs'][0]['text']
        ex_prompt = f"{title}\n\n{text}\n\nBased on this text, answer these questions:\nQ: {q}\nA:"
    else:
        q = normalize_question(example["question"])
        if example["ctxs"][0]["title"] is not None:
            docs_text = "\n\n".join([f"{ctx['title']}\n\n{ctx['text']}" for ctx in example["ctxs"][:num_docs]])
        else:
            docs_text = "\n\n".join([f"Document {i}: {ctx['text']}" for i, ctx in enumerate(example["ctxs"][:num_docs])])
        if require_long:
            ex_prompt = f"{docs_text}\n\nBased on these texts, answer these questions in full sentence, as completely as possible:\nQ: {q}\nA:"
        else:
            ex_prompt = f"{docs_text}\n\nBased on these texts, answer these questions:\nQ: {q}\nA:"

    return ex_prompt
