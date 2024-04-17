#!/bin/bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

# Reload bash shell configuration to recognize Conda
source ~/.bashrc

# Create new Conda environment with Python 3.12
~/miniconda3/bin/conda create --name week7day1 python=3.12 -y

echo Attempting to print the environment
# Activate the new environment
source ~/miniconda3/bin/activate week7day1

echo Environment set to:

~/miniconda3/bin/conda info --envs
#conda install cudatoolkit=11.0 -y
pip install transformers datasets peft accelerate bitsandbytes
