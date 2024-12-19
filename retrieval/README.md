# Retrieval Tutorial

This guide explains how to perform retrieval using the Pyserini library. Depending on your use case (retrieving from a local index or the MSMARCOv2 index), you’ll use either `retrieve_local.sh` or `retrieve_ms2.sh`.

## Why Pyserini?
Pyserini simplifies retrieval by providing prebuilt indices. It eliminates the need for training your retrieval models, allowing you to focus on using and analyzing retrieval results. After obtaining retrieval results, you can enhance them by adding raw text and other metadata.

---

## Prerequisites

1. **Install Required Libraries:**
   - **FAISS and PyTorch:**
    See [here]("https://github.com/facebookresearch/faiss/blob/main/INSTALL.md") for more reference. The below example is for CUDA 11.4.
     ```bash
     conda create --name faiss_1.7.4 python=3.10
     conda activate faiss_1.7.4
     conda install faiss-gpu=1.7.4 mkl=2021 pytorch pytorch-cuda numpy -c pytorch -c nvidia
     ```
   - **Java (for Lucene):**
     ```bash
     conda install -c conda-forge openjdk=11 maven -y
     ```
   - **Pyserini:**
     ```bash
     pip install pyserini
     ```

2. **Set Environment Variables:**  
   Add the path to `libjvm.so` if you encounter `RuntimeError: Unable to find libjvm.so`:
   ```bash
   export JAVA_HOME=/path/to/java/home
   ```

3. **Input Dataset Format:**
   The input dataset file should be a JSON file, which is a list of dictionaries. Each dictionary must have a key `"question"`, where the value is the query to be searched for. For example:
   ```json
   [
       {"question": "What is the capital of France?", "other_key": "value"},
       {"question": "Who won the first Nobel Prize in Physics?", "other_key": "value"}
   ]
   ```
---

## Workflow Overview
1. **Local Index Retrieval:** Use `retrieve_local.sh` for custom datasets and local FAISS indices.
2. **MSMARCOv2 Index Retrieval:** Use `retrieve_ms2.sh` for prebuilt MSMARCOv2 Lucene indices.

---

## Local Index Retrieval

Use `retrieve_local.sh` to retrieve results from a **local FAISS index** for custom datasets.

### Example Command:
```bash
bash retrieve_local.sh <corpus_name> <dataset_type> <dataset_path> <dataset_name>
```

### Parameters:
- **`<corpus_name>`**: Index name (e.g., `wiki`, `web`, or `wiki-web`).
- **`<dataset_type>`**: Dataset source (`local` or `pyserini`).
- **`<dataset_path>`**: Path to the dataset (for local datasets).
- **`<dataset_name>`**: Name of the dataset (e.g., `msmarcoqa`).

### Example Usage:
```bash
bash retrieve_local.sh wiki local ../data/custom_dataset.json custom-dataset
```

This script:
1. Converts the raw dataset to a topics file.
2. Runs FAISS retrieval on the local index.
3. Converts retrieval results to DPR format.

---

## MSMARCOv2 Index Retrieval

Use `retrieve_ms2.sh` to retrieve results from the **MSMARCOv2 Lucene index**.

### Example Command:
```bash
bash retrieve_ms2.sh <dataset_short_name>
```

### Parameters:
- **`<dataset_short_name>`**: Name of the dataset (e.g., `nq-test`, `msmarcoqa`, `hotpot`).

### Example Usage:
```bash
bash retrieve_ms2.sh nq-test
```

This script:
1. Converts the dataset to a topics file.
2. Performs BM25 retrieval using the MSMARCOv2 Lucene index.
3. Converts the retrieval results to DPR format.

---

## Troubleshooting

- **`RuntimeError: Unable to find libjvm.so`**:
  Ensure Java is installed and `JAVA_HOME` is set.
- **Missing files or directories**: Verify that all dataset and index paths are correct.
