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
hostmachine=`hostname`
if [[ $hostmachine == 'cloud-vm-45-110' ]]; then
  ## Path on the server!
  export NLTK_DATA="/usr/local/share/nltk_data"
else
  ## Path on local machines
  export NLTK_DATA="$HOME/nltk_data"
fi
cwd=$(pwd)
cd $NLTK_DATA

# One time install run for stanford
# wget http://nlp.stanford.edu/software/stanford-ner-2015-04-20.zip
# curl http://nlp.stanford.edu/software/stanford-ner-2015-04-20.zip > stanford-ner.zip
# unzip stanford-ner.zip -d stanford-ner
# rm -f stanford-ner.zip

export ST="$NLTK_DATA/stanford-ner"
export CLASSPATH="$ST/stanford-ner.jar:$ST/lib/joda-time.jar:$ST/lib/jollyday-0.4.7.jar:$ST/lib/stanford-ner-resources.jar"
export STANFORD_MODELS="$ST/classifiers"

## download nltk trainer and prepare the NLTK taggers and chunkers used by seek
## warning: slow
#git clone https://github.com/japerk/nltk-trainer
#cd nltk-trainer
#python setup.py install
#python train_chunker.py treebank
#python train_tagger.py brown

cd $cwd
