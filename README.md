# RAG with Passage Combination

This repository provides tools and scripts to run retrieval and train a model to retrieve passage combinations that helps downstream retrieval-augmented QA the most. A baseline is also provided by simply retrieving and taking the top k passages.

## Usage

1. **Run Retrieval**: 
   - Use the provided scripts indicated in [retrieval README](retrieval/README.md) to run the retrieval process. This step involves generating representation vectors for the static documents dataset and retrieving the best matching passages given the query vectors.

2. **Train Model**:
   - Train a model to retrieve passage combinations using the provided `.sh` files.
   ```bash
   bash qa_passage_combination/train_and_evaluate.sh
   ```

3. **Evaluate Model**:
   - Evaluate the trained model using the provided `.sh` files.
   ```bash
   bash qa_passage_combination/test.sh
   ```

4. **Baseline**:
   - A baseline is provided by simply retrieving and taking the top k passages. This can be used as a reference to compare the performance of your trained model.
   ```bash
   bash qa_baseline/rerank.sh
   bash qa_baseline/run_qa_all.sh
   ```

## Acknowledgement

This project is based on and inspired by the work and code from the following repositories:

1. [Hannibal046/nanoDPR](https://github.com/Hannibal046/nanoDPR)
2. [castorini/pyserini](https://github.com/castorini/pyserini)
3. [AI21Labs/in-context-ralm](https://github.com/AI21Labs/in-context-ralm)
4. [StonyBrookNLP/ircot](https://github.com/StonyBrookNLP/ircot)

We thank the authors of these repositories for their contributions to the community.


## License

This project is licensed under the MIT License.
