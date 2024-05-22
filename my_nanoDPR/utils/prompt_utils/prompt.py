# %%

# llama prompt src
# https://github.com/IntelLabs/fastRAG/blob/3959edd43c1f69bf028e48992c5c78bdd65002f1/fastrag/rest_api/chat_template_initalizers.py#L4
# https://github.com/IntelLabs/fastRAG/blob/main/examples/replug_parallel_reader.ipynb

# QA prompt src
# https://github.com/IntelLabs/fastRAG/blob/3959edd43c1f69bf028e48992c5c78bdd65002f1/config/rag_generation_with_dynamic_prompt.yaml#L28
# "Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words. \n\n Context: {join(documents)} \n\n Question: {query} \n\n Answer:"

# flan prompt src
# https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py

# %%
# llama_2_prompt_no_doc = """[INST] <<SYS>>
# Answer the Question below considering the Document provided.
# Your answer can only be an entity name or a short phrase.

# Examples:
# {exemplars}

# <</SYS>>

# Question: {question}
# Answer: [/INST]
# """

# llama_2_prompt_with_doc = """[INST] <<SYS>>
# Answer the Question below considering the Document provided.
# Your answer can only be an entity name or a short phrase.

# Examples:
# {exemplars}

# Document: {documents}
# <</SYS>>

# Question: {question}
# Answer: [/INST]
# """

# if we're using llama1, then FEW-SHOT WON'T WORK!!!!!!
# llama_prompt_no_doc = """{exemplars}

# Answer these questions:
# Q: {question}
# A:
# """
# %%
from utils.prompt_utils.nq_shots import get_nq_exemplars

"""for llama3"""
# my mimic of the apply_chat_template
chat_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYS_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{USER_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3
llama3_sys_prompt_no_doc = """You are an assistant for directly answering questions in a compact way.
Provide a very short answer that contains only necessary entities.
For example:
Question: Who is the president of the United States?
Answer: Joe Biden"""
llama3_sys_prompt_with_doc = """You are an assistant for directly answering questions in a compact way.
You are given the extracted parts of a long document and a question. Provide a very short answer that contains only necessary entities.
For example:
Question: Who is the president of the United States?
Context: Recently, Joe Biden was elected as the president of the United States.
Answer: Joe Biden"""

def format_llama3_user_prompt(question,retrieved_documents,k):
    """using the retrieved documents we will prompt the model to generate our responses"""
    USER_PROMPT = f"Question:{question}\n"
    if retrieved_documents == []:
        return USER_PROMPT + "Answer:"
    USER_PROMPT+= "Context:"
    for idx in range(k) :
        USER_PROMPT+= f"{retrieved_documents[idx]}\n"
    USER_PROMPT+= "Answer:"
    return USER_PROMPT


"""for llama1 and flan"""
llama_prompt_no_doc = "Answer these questions:\nQuestion: {question}\nAnswer:"
llama_prompt_with_doc = "{exemplars}\n\nDocument: {documents}\nBased on these texts, answer these questions:\nQuestion: {question}\nAnswer:"

flan_prompt_no_doc = "Give an answer to the answerable question.\n\n{exemplars}\n\nQuestion: {question}\nAnswer:"
flan_prompt_with_doc = "Give an answer to the question.\n\n{exemplars}\n\nContext: {documents}\nQuestion: {question}\nAnswer:"

prompt_collection = {
    "llama": {
        "no_doc": llama_prompt_no_doc,
        "with_doc": llama_prompt_with_doc
    },
    "flan": {
        "no_doc": flan_prompt_no_doc,
        "with_doc": flan_prompt_with_doc
    }
}

def make_prompt(question, documents, lm_name, num_exemplars, dataset):
    assert isinstance(documents, list), "When make_prompt(), documents should be a list of strings."
    documents = [doc for doc in documents if doc != ""]
    num_docs = len(documents)

    if "llama-3" in lm_name.lower():
        lm_name = "llama3"
        if num_exemplars != 1:
            raise ValueError("When make_prompt(), llama-3 should always use num_exemplars=1.")
    elif "llama" in lm_name:
        lm_name = "llama"
    elif "flan" in lm_name:
        lm_name = "flan"
    elif "gpt2" in lm_name:
        lm_name = "flan" # tmp hack
    else:
        raise ValueError("lm_name only support llama or flan now.")
    
    if lm_name == "llama3":
        # raise NotImplementedError("llama3 is not implemented yet.")
        formatted_prompt = format_llama3_user_prompt(question, documents, num_docs)
        sys_prompt = llama3_sys_prompt_no_doc if num_docs == 0 else llama3_sys_prompt_with_doc
        # remove the ending newline if any
        sys_prompt, formatted_prompt = sys_prompt.strip(), formatted_prompt.strip()
        return chat_prompt_template.format(SYS_PROMPT=sys_prompt, USER_PROMPT=formatted_prompt)
    
    if dataset == "nq":
        exemplars = get_nq_exemplars(lm_name, num_docs, num_exemplars)
    else:
        raise ValueError("dataset only support nq now.")
    
    if num_docs == 0:
        return prompt_collection[lm_name]["no_doc"].format(
            exemplars="\n\n".join(exemplars), question=question
        )
    else:
        return prompt_collection[lm_name]["with_doc"].format(
            exemplars="\n\n".join(exemplars), question=question, 
            documents="\n".join(documents)
        )

# %%