import torch
from transformers import AutoConfig, AutoTokenizer


def load_query_encoder_and_tokenizer(args, logger):
    logger.info(f"...Loading query encoder model from {args.query_encoder}...")
    if "dpr" == args.encoder_type:
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.query_encoder)
        query_encoder = DPRQuestionEncoder.from_pretrained(args.query_encoder)
    else:
        from transformers import BertModel, BertTokenizer
        query_tokenizer = BertTokenizer.from_pretrained(args.query_encoder)
        query_encoder = BertModel.from_pretrained(args.query_encoder,add_pooling_layer=False)
    logger.debug(f"GPU memory used: {torch.cuda.memory_allocated() / 1e6} MB")
    return query_tokenizer, query_encoder


def load_doc_encoder_and_tokenizer(args, logger):
    logger.info(f"...Preparing doc encoder...")
    if "dpr" == args.encoder_type:
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        doc_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.doc_encoder)
        doc_encoder = DPRContextEncoder.from_pretrained(args.doc_encoder)
    else:
        from transformers import BertTokenizer, BertModel
        doc_tokenizer = BertTokenizer.from_pretrained(args.doc_encoder)
        doc_encoder = BertModel.from_pretrained(args.doc_encoder,add_pooling_layer=False)
    doc_encoder.eval()
    return doc_tokenizer, doc_encoder


def load_lm_tokenizer(model_name):
    if "llama" in model_name:
        from transformers import LlamaTokenizer
        lm_tokenizer = LlamaTokenizer.from_pretrained(model_name)
        lm_tokenizer.pad_token = "[PAD]"
        lm_tokenizer.padding_side = "left"
        return lm_tokenizer
        # return LlamaTokenizer.from_pretrained(model_name, padding_side="left", pad_token="[PAD]") # cause CUDA error
    return AutoTokenizer.from_pretrained(model_name)


def load_lm_model_and_tokenizer(model_name, device, model_parallelism=False, cache_dir=None, auth_token=None):
    device_count = torch.cuda.device_count()
    print("Using device: ", device)
    if type(device) == str:
        assert 'cuda' in device, f"CPU not supported!!!! You are using {device} device."
    else:
        assert 'cuda' in device.type, f"CPU not supported!!!! You are using {device.type} device."

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

    # TODO 感覺會跟 accelerator 有衝突
    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, config

