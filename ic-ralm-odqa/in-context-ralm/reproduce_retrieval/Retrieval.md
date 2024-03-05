# Retrieval Tutorial
We're going to use DPR retrieval by Pyserini library. The benifit of using Pyserini lies in the retrieval index it precomputes. Since we can directly use the index, we don't need to train DPR by ourselves.
After we get retrieval results from Pyserini, we still need to run a little program to get the raw text of each context.

This whole retrieval (1st+2nd stage) can be run by `generate_retrieval.sh` by setting proper variable names.

## Pyserini Installation
Install faiss and pytorch together first, see [here]("https://github.com/facebookresearch/faiss/blob/main/INSTALL.md").
In short, since I have CUDA 11.4, I use this directly
```
conda create --name faiss_1.7.4 python=3.10
conda activate faiss_1.7.4
conda install faiss-gpu=1.7.4 mkl=2021 pytorch pytorch-cuda numpy -c pytorch -c nvidia
```

Install JAVA
```
conda install -c conda-forge openjdk=11 maven -y
```

Then, install pyserini.
```
pip install pyserini
```
Remember to follow the guide and test it!

### Troubleshoot
`RuntimeError: Unable to find libjvm.so`: Probably you forgot to install JAVA! See the instruction above.

Or, try the following
```
echo $JAVA_HOME
export JAVA_HOME=/path/to/java/home
```

## 1st stage Retrieval (Run Pyserini)
In my case, first move to the `~/research/ic-ralm-odqa/in-context-ralm/reproduce_retrieval`

To reproduce DPR results, visit [here]("https://github.com/castorini/pyserini/blob/master/docs/experiments-dpr.md")
Or, [here]("https://castorini.github.io/pyserini/2cr/odqa.html") offers complete reproduction commands.

### NQ 
First, generate a `run.odqa.DPR.nq-test.hits-100.txt` file. 
* `--index` can be the custom **DPR** index.
* Change `--output` to be the result destination.

```
python -m pyserini.search.faiss \
  --threads 16 --batch-size 512 \
  --index wikipedia-dpr-100w.dpr-single-nq \
  --encoder facebook/dpr-question_encoder-single-nq-base \
  --topics nq-test \
  --output run.odqa.DPR.nq-test.hits-100.txt \
  --hits 100
```

Then, convert this file to json format.
* `--index` should be the **Lucene** index for raw text correspondance.
* Change `--input` and `--output` as custom path.
* `--store-raw` will add a "text" field 


```
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
  --topics nq-test \
  --index wikipedia-dpr \
  --input run.odqa.DPR.nq-test.hits-100.txt \
  --output run.odqa.DPR.nq-test.hits-100.json
  [--store-raw]
```

Now, we should already get 100 relevant passage for each question, with docid, score, and has_answer.

Optionally, perform evaluation on retrieval results (recall, top-k).
```
python -m pyserini.eval.evaluate_dpr_retrieval \
  --retrieval run.odqa.DPR.nq-test.hits-100.json \
  --topk 20 100
```

## 2nd stage Retrieval (Raw Text Supplement)
Now, we want to put the raw text into the json file so that each context has id, title, text, score, and has_answer.
(If `--store-raw` is set when converting `txt` to `json` file at the last step, this step is of no need.)

<!-- ```
python format.py \
    --faiss_index_dir ~/research/data_augmentation/faiss_index/all \
    --lucene_index_dir ~/research/data_augmentation/lucene_index_orig_para1 \
    --no_raw_text_file result/myindex.nq-test.hits-100.json \
``` -->
```
python format.py \
    --input_file result/msmarco-unicoil.nq-test.hits-100.json \
    --output_file result/formatted-msmarco-unicoil.nq-test.hits-100.json \
    --have_raw
```
