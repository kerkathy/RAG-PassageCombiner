import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
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
    input_ids,
    attention_mask,
    token_type_ids,
    max_length, max_tokens_to_generate, 
    num_orig_question, temperature=1
):
    """
    Take a model and a batch of input_ids, attention_mask, and token_type_ids and return the log probability of the input_ids
    The return value is NOT a distribution, but a log probability
    Note: ext_batch_size here = num_orig_question * n_comb
    """
    if input_ids.shape[-1] > max_length - max_tokens_to_generate:
        num_too_long += 1
        input_ids = input_ids[..., -(max_length - max_tokens_to_generate):]

    outputs = model(input_ids, attention_mask=attention_mask).logits
    probs = torch.log_softmax(outputs, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids, token_type_ids = input_ids[:, 1:], token_type_ids[:, 1:]
    probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1) # [ext_batch_size, seq_len, 1] -> [ext_batch_size, seq_len]
    
    # set the log probs to 0 where token_type_ids = 0
    probs[token_type_ids == 0] = 0 # [ext_batch_size, seq_len]

    # compute sequence scores
    # option 1. sum of log probabilities 
    sum_log_probs = probs.sum(dim=-1).view(num_orig_question, -1) # TODO: exp or not
    # option 2. joint probability
    # joint_probs = probs.sum(dim=-1).view(num_orig_question, -1).exp() # [num_orig_question, n_comb]

    # %%
    # Turn n_comb into a distribution
    # sum_log_probs = torch.softmax(sum_log_probs / temperature, dim=-1).detach() # [num_orig_question, n_comb]
    # joint_probs = torch.softmax(joint_probs / temperature, dim=-1).detach() # [num_orig_question, n_comb]
    
    # %%
    return sum_log_probs # [num_orig_question, n_comb]
