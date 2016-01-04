#!/bin/bash

# virtualenv -p /usr/local/bin/python3 seek-env

source seek-env/bin/activate

cd seek-env

git clone https://github.com/japerk/nltk-trainer
cd nltk-trainer
python setup.py install
python train_chunker.py treebank
python train_tagger.py brown

# pip3 install -U numpy
# pip3 install -U scipy
# pip3 install -U gensim
# pip3 install -U nltk
# python -m nltk.downloader all
