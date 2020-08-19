

1. Please download and extract selfanime data under folder dataset

2. Assume you extract the self2anime dataset under path "dataset/bundle"


## Run a full model with gpu
python main.py --device=cuda --dataset=bundle


## Run a light model with gpu
python main.py --device=cuda --light True --dataset=bundle
