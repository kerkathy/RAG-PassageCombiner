# %%

# llama prompt src
# https://github.com/IntelLabs/fastRAG/blob/3959edd43c1f69bf028e48992c5c78bdd65002f1/fastrag/rest_api/chat_template_initalizers.py#L4
# https://github.com/IntelLabs/fastRAG/blob/main/examples/replug_parallel_reader.ipynb

# QA prompt src
# https://github.com/IntelLabs/fastRAG/blob/3959edd43c1f69bf028e48992c5c78bdd65002f1/config/rag_generation_with_dynamic_prompt.yaml#L28
# "Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words. \n\n Context: {join(documents)} \n\n Question: {query} \n\n Answer:"

# flan prompt src
# https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py


from utils.prompt_utils.nq_shots import get_nq_exemplars

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
llama_prompt_no_doc = """Answer these questions:
Q: {question}
A:"""

# llama_prompt_with_doc = """{exemplars}

# {documents}
# Based on these texts, answer these questions:
# Q: {question}
# A:
# """
llama_prompt_with_doc = """{documents}

Based on these texts, answer these questions:
Q: {question}
A:"""

flan_prompt_no_doc = """Give an answer to the answerable question.
{exemplars}

Question: {question}
Answer:"""

flan_prompt_with_doc = """Give an answer to the question.
{exemplars}

Context: {documents}
Question: {question}
Answer:"""

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

def make_prompt(question, documents, lm_name, num_docs, num_exemplars, dataset):
    if "llama" in lm_name:
        lm_name = "llama"
    elif "flan" in lm_name:
        lm_name = "flan"
    elif "gpt2" in lm_name:
        lm_name = "flan" # tmp hack
    else:
        raise ValueError("lm_name only support llama or flan now.")
    
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