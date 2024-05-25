import torch
from transformers import AutoConfig, AutoTokenizer


def load_query_encoder_and_tokenizer(args, logger):
    logger.info(f"...Loading query encoder model from {args.query_encoder}...")
    if "dpr" == args.query_encoder_type:
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
    if "dpr" == args.doc_encoder_type:
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
    if "llama-7b" in model_name:
        from transformers import LlamaTokenizer
        lm_tokenizer = LlamaTokenizer.from_pretrained(model_name)
        # lm_tokenizer.pad_token = "[PAD]"
        # lm_tokenizer.padding_side = "left" # it's no use setting this here. Set it in main insead
        return lm_tokenizer
        # return LlamaTokenizer.from_pretrained(model_name, padding_side="left", pad_token="[PAD]") # cause CUDA error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if "llama-3" in model_name.lower():
    #     print("Using LLAMA-3 tokenizer")
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

# TODO 改這裡成跟 accelerate 兼容的樣子
# https://github.com/huggingface/accelerate/issues/2163
# https://huggingface.co/docs/accelerate/concept_guides/big_model_inference
# https://huggingface.co/docs/accelerate/usage_guides/big_modeling

def load_lm_model_and_tokenizer(model_name, device=None, quantized=False, model_parallelism=False, cache_dir=None, auth_token=None):
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
    tokenizer = load_lm_tokenizer(model_name)
    config = AutoConfig.from_pretrained(model_name)

    if not quantized:
        device_count = torch.cuda.device_count()
        print("Using device: ", device)
        if type(device) == str:
            assert 'cuda' in device, f"CPU not supported!!!! You are using {device} device."
        else:
            assert 'cuda' in device.type, f"CPU not supported!!!! You are using {device.type} device."

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
    
    elif not model_parallelism:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir, quantization_config=quantization_config)

    elif model_parallelism:
        raise NotImplementedError("model_parallelism and quantized cannot be used together (for now)")
        # https://huggingface.co/docs/accelerate/concept_guides/big_model_inference
        # https://huggingface.co/docs/accelerate/v0.30.1/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch
        # assert "llama" in model_name.lower(), "model_parallelism only support llama models (for now)"

        # from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        # from huggingface_hub import hf_hub_download

        # # Download the Weights
        # weights_location = hf_hub_download(model_name, f"{cache_dir}/{model_ckpt_path}")

        # # Create a model and initialize it with empty weights
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config=config)

        # # Load the checkpoint and dispatch it to the right devices
        # model = load_checkpoint_and_dispatch(
        #     model, weights_location, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
        # )

        # print("Model MEM footprint", model.get_memory_footprint())
        # print("Model device", model.device)


    return model, tokenizer, config