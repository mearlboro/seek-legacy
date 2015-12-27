#!/bin/bash

virtualenv -p /usr/local/bin/python3 seek-env

/bin/bash -c ". seek-env/bin/activate; exec /bin/bash -i"
cd seek-env

pip3 install -U numpy
pip3 install -U scipy
pip3 install -U gensim
pip3 install -U nltk

python -m nltk.downloader all 
