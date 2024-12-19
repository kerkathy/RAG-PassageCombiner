import torch
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer


def load_tokenizer(model_name):
    if "llama" in model_name:
        return LlamaTokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def load_model_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None):
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
    tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, config, device
