Reminder for options that should be set together
* `query_encoder: google-bert/bert-base-uncased`: `doc_encoder_type: bert`, `query_encoder_type: bert`
* `query_encoder: facebook/contriever`: `doc_encoder_type: contriever`, `query_encoder_type: contriever`
* `query_encoder: facebook/dpr-question_encoder-multiset-base`: `doc_encoder_type: dpr-multiset`, `query_encoder_type: dpr-multiset`
* `query_encoder: facebook/dpr-question_encoder-single-nq-base`: `doc_encoder_type: dpr`, `query_encoder_type: dpr`

To run a new test to see the performance of the checkpoint of a specific step in a run saved in wandb, copy either `test_dpr_<dataset>.yaml` (`<dataset>` can be `hotpot`, `nq`, `trivia`) to a new file. Then, modify below info:
* `runid_to_eval`: the wandb run id of the run
* `eval_steps`: the step of the certain checkpoint in that run
* `max_eval_steps`: total number of steps in that run
* `max_round`: the number of retrieval rounds in that train run (or eval? I forgot...) 
