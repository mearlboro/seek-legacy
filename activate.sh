#!/bin/bash

# the purpose of this script is to update the python virtual environment after
# various updates and new packages have been installed and avoid running the
# heavy install.sh script. please comment out duplicate items as some are very
# resource-intensive and involve ML training

# activates environment
source seek-env/bin/activate

## -- Gensim -------------------------------------------------------------------
#pip3 install -U numpy
#pip3 install -U scipy
#pip3 install -U regex
#pip3 install -U gensim

## -- NLTK ---------------------------------------------------------------------
#pip3 install -U nltk
#python -m nltk.downloader all

## prepare NLTK data and download stanford NLP package
## Path on local machines
# export NLTK_DATA="$HOME/nltk_data"
## Path on the server!
#export NLTK_DATA="/usr/local/share/nltk_data"
cwd=$(pwd)
cd $NLTK_DATA

# wget http://nlp.stanford.edu/software/stanford-ner-2014-06-16.zip
curl http://nlp.stanford.edu/software/stanford-ner-2014-06-16.zip > stanford-ner.zip
unzip stanford-ner.zip -d stanford-ner
rm -f stanford-ner.zip
export CLASSPATH="$NLTK_DATA/stanford-ner/stanford-ner.jar"
export STANFORD_MODELS="$NLTK_DATA/stanford-ner/classifiers"

## download nltk trainer and prepare the NLTK taggers and chunkers used by seek
## warning: slow
#git clone https://github.com/japerk/nltk-trainer
#cd nltk-trainer
#python setup.py install
#python train_chunker.py treebank
#python train_tagger.py brown

cd $cwd
