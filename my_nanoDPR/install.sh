set -e
# install pytorch according to driver version
# cuda 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
# other version
# ...

conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
# [stable, workable]
# pip install transformers==4.30.2 accelerate==0.20.3 wandb wget spacy
# [experimental]
pip install transformers accelerate bitsandbytes>0.37.0 wandb wget spacy
conda install conda-forge::sentencepiece -y


# may need to do this on a cu113 version
# pip install bitsandbytes-cuda116