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
    import yaml  
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
    import torch
    inputs = tokenizer(doc, return_tensors='pt', truncation=True, padding=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Moving inputs to device: ", device)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    print("inputs.keys(): ", inputs.keys())
    print("inputs['input_ids'].shape: ", inputs['input_ids'].shape)
    print("inputs['attention_mask'].shape: ", inputs['attention_mask'].shape)
    print("inputs['token_type_ids'].shape: ", inputs['token_type_ids'].shape)
    print("...Sending into model...")
    # TODO debug here!!
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def make_index(corpus, tokenizer, encoder, batch_size=32):
    """
    input
        * corpus (List[Str]): List of documents
        * doc_encoder (Callable): Document encoder function
    return
        * doc_embeddings (torch.Tensor): Document embeddings
    """
    import torch

    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = [normalize_document(doc) for doc in corpus[i:i+batch_size]]
            inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True)
            outputs = encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings_list.extend(embeddings.tolist())
    return torch.tensor(embeddings_list)

def retrieve_top_k_docid(query, doc_embeddings, tokenizer, query_encoder, k):  # Add the device argument
    """
    Compute cosine similarity between query and documents in the doc_index
    return top k documents
    """
    import torch

    query_embedding = get_sentence_embedding(query, tokenizer, query_encoder)  # Pass the device to get_sentence_embedding
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_embedding, doc_embeddings = query_embedding.to(device), doc_embeddings.to(device)
    scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings, dim=1)
    top_doc_scores, top_doc_indices = torch.topk(scores, k)
    top_doc_indices = top_doc_indices.flatten().tolist()

    return top_doc_indices
