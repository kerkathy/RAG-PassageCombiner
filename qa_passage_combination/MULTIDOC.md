# Multi-document Retrieval
## Objective
In this repo, we train a model to find a optimal permutation of documents for a given question.
The retriever consist of two BERT encoder, one for question and the other for context. The former is trainable while the latter is fixed.

## Preparation
- `train.json`: Multiple (question, answer) pairs
- `ctx.json`: Raw documents, (id, text) pairs
- `encoded_ctx.json`: Document embeddings, (id, embedding) pairs. Or faiss index
    <!-- - Should be updated every 3k steps -->

## Training
```
python train.py \
    --train_datasets <train_file>
    --eval_datasets <eval_file>
    --output_dir <path to ckpts>
    --ctx <path to raw doc>
    --pretrained_q_enc <path to pretrained doc encoder ckpt>
    --pretrained_ctx_enc <path to pretrained doc encoder ckpt>
```



## Evaluation


## Inference

