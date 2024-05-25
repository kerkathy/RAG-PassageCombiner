import torch
import yaml,os
def ensure_directory_exists_for_file(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def set_seed(seed: int = 19980406):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def normalize_document(document: str):
    document = document.replace("\n", " ").replace("’", "'")
    if document.startswith('"'):
        document = document[1:]
    if document.endswith('"'):
        document = document[:-1]
    return document

def normalize_query(question: str) -> str:
    question = question.replace("’", "'")
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def get_yaml_file(file_path):
    with open(file_path, "r") as file:  
        config = yaml.safe_load(file)  
    return config  

def get_linear_scheduler(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_sentence_embedding(doc, tokenizer, model):
    inputs = tokenizer(doc, return_tensors='pt', truncation=True, padding=True)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    outputs = model(**inputs)
    if "dpr" == model.config.model_type:
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def make_index(corpus, tokenizer, encoder, batch_size=32):
    """
    input
        * corpus (List[Str]): List of documents
        * doc_encoder (Callable): Document encoder function
    return
        * doc_embeddings (torch.Tensor): Document embeddings
    """
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = [normalize_document(doc) for doc in corpus[i:i+batch_size]]
            embeddings = get_sentence_embedding(batch, tokenizer, encoder)
            embeddings_list.extend(embeddings.tolist())
    return torch.tensor(embeddings_list)

def retrieve_top_k_docid(query, doc_embeddings, tokenizer, query_encoder, k):
    """
    Compute cosine similarity between query and documents in the doc_index
    return top k documents
    """
    # if query in cache:
    #     print("Cache hit!")
    #     return cache[query]
    query_embedding = get_sentence_embedding(query, tokenizer, query_encoder)  
    # query_embedding, doc_embeddings = query_embedding.to(device), doc_embeddings.to(device)
    # query_embedding, doc_embeddings = accelerator.prepare(query_embedding), accelerator.prepare(doc_embeddings) # this line cause error: device mismatch with one on cpu and one on cuda
    # print(f"query_embedding at {query_embedding.device}; doc_embeddings at {doc_embeddings.device}")
    # query_embedding = query_embedding.to(doc_embeddings.device)
    doc_embeddings = doc_embeddings.to(query_embedding.device)
    scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings, dim=1)
    top_doc_scores, top_doc_indices = torch.topk(scores, k)
    top_doc_indices = top_doc_indices.flatten().tolist()
    # cache[query] = top_doc_indices

    return top_doc_indices