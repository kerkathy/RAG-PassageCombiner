set -e
# install pytorch according to driver version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
# [stable, workable]
# pip install transformers==4.30.2 accelerate==0.20.3 wandb wget spacy
# [experimental]
pip install transformers accelerate bitsandbytes>0.37.0
conda install conda-forge::sentencepiece -y