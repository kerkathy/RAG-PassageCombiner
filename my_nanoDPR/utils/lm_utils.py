import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import login

def load_lm_tokenizer(model_name):
    if "llama" in model_name:
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

    # model = model.to(device)
    # print("LM model device: ", model.device)
    # print("input_ids device: ", input_ids.device)
    # print("attention_mask device: ", attention_mask.device)
    # print("token_type_ids device: ", token_type_ids.device)
    all_outputs = []
    for input_ids_batch, attention_mask_batch, token_type_ids_batch in zip(input_ids.split(llm_batch_size), attention_mask.split(llm_batch_size), token_type_ids.split(llm_batch_size)):
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        token_type_ids_batch = token_type_ids_batch.to(device)
        with torch.no_grad():
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch).logits
            outputs = torch.log_softmax(outputs, dim=-1).detach()
            all_outputs.append(outputs)
    outputs = torch.cat(all_outputs, dim=0)
    
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    outputs = outputs[:, :-1, :]
    input_ids, token_type_ids = input_ids[:, 1:], token_type_ids[:, 1:]
    outputs = torch.gather(outputs, 2, input_ids[:, :, None]).squeeze(-1) # [ext_batch_size, seq_len, 1] -> [ext_batch_size, seq_len]
    
    # set the log probs to 0 where token_type_ids = 0
    outputs[token_type_ids == 0] = 0 # [ext_batch_size, seq_len]

    # compute sequence scores
    # option 1. sum of neg log probabilities 
    outputs = outputs.sum(dim=-1).view(num_orig_question, -1) # TODO: exp or not
    # option 2. joint probability
    # joint_probs = probs.sum(dim=-1).view(num_orig_question, -1).exp() # [num_orig_question, n_comb]

    return outputs # [num_orig_question, n_comb]
