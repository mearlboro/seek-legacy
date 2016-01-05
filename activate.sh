#!/bin/bash

# virtualenv -p /usr/local/bin/python3 seek-env

source seek-env/bin/activate

# export NLTK_DATA="$HOME/nltk_data"
export NLTK_DATA="/usr/local/share/nltk_data"
cd $NLTK_DATA

# wget http://nlp.stanford.edu/software/stanford-ner-2014-06-16.zip
git clone https://github.com/japerk/nltk-trainer
cd nltk-trainer
python setup.py install
# cd seek-env
export CLASSPATH="$NLTK_DATA/stanford-ner-2014-06-16/stanford-ner.jar"
export STANFORD_MODELS="$NLTK_DATA/stanford-ner-2014-06-16/classifiers"

# git clone https://github.com/japerk/nltk-trainer
# cd nltk-trainer
# python setup.py install
# python train_chunker.py treebank
# python train_tagger.py brown

# pip3 install -U numpy
# pip3 install -U regex
# pip3 install -U scipy
# pip3 install -U gensim
# pip3 install -U nltk
# python -m nltk.downloader all
