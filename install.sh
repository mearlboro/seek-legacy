#!/bin/bash

platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
  platform='osx'
fi


# -- required: curl -------------------------------------------------------------
if [[ $platform == 'linux' ]]; then
  sudo apt-get install curl
elif [[ $platform == 'osx' ]]; then
  brew install curl
fi

# -- ruby -----------------------------------------------------------------------
# gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
curl -sSL https://get.rvm.io | bash -s stable --ruby
rvm install 2.2.1
/bin/bash --login
rvm use 2.2.1
sudo gem install mechanize

if [[ $platform == 'linux' ]]; then
  # -- python -------------------------------------------------------------------
  sudo apt-get install python3
  sudo apt-get install python-virtualenv
  sudo apt-get install python3-pip

  # -- setup environment --------------------------------------------------------
  virtualenv -p /usr/bin/python3 seek-env
  # -- dont forget to always activate / deactivate ------------------------------
  /bin/bash -c ". seek-env/bin/activate; exec /bin/bash -i"
  cd seek-env

  # -- textract dependencies ----------------------------------------------------
  sudo apt-get install python3-dev libxml2 antiword unrtf poppler-utils pstotext tesseract-ocr flac lame libmad0 libsox-fmt-mp3 sox libjpeg-dev zlib1g-dev

elif [[ $platform == 'osx' ]]; then
  # -- python -------------------------------------------------------------------
  brew install python3
  pip3 install virtualenv

  # -- setup environment --------------------------------------------------------
  virtualenv -p /usr/local/bin/python3 seek-env
  # -- dont forget to always activate / deactivate ------------------------------
  /bin/bash -c ". seek-env/bin/activate; exec /bin/bash -i"
  cd seek-env

  # -- textract dependencies ----------------------------------------------------
  brew install libxml2 antiword unrtf poppler tesseract flac lame libmad libsoxr libjpeg
  brew install homebrew/dupes/zlib

fi

# -- setup textract: continue ---------------------------------------------------
# installer script based on http://www.tysonmaly.com/installing-textract-for-python-3/
pip3 install --upgrade lxml
curl https://pypi.python.org/packages/source/t/textract/textract-1.4.0.tar.gz > textract-1.4.0.tar.gz
tar xvf textract-1.4.0.tar.gz
cd textract-1.4.0
ed -s requirements/python <<< $',s/pdfminer==20140328/\nw'
pip3 install pdfminer3k
python3 setup.py install
cd ..
ed -s lib/python3.4/site-packages/textract-1.4.0-py3.4.egg/textract/parsers/utils.py <<< $',s/(text, unicode)/(text, str)\nw'


# -- NLTK -----------------------------------------------------------------------
pip3 install -U nltk
python -m nltk.downloader all

# prepare nltk data and download stanford NLP package 
# Path on local machines
export NLTK_DATA="$HOME/nltk_data"
## Path on the server!
#export NLTK_DATA="/usr/local/share/nltk_data"
cwd=$(pwd)
cd $NLTK_DATA

curl http://nlp.stanford.edu/software/stanford-ner-2014-06-16.zip > stanford-ner.zip
unzip stanford-ner.zip -d stanford-ner
rm -f stanford-ner.zip
export CLASSPATH="$NLTK_DATA/stanford-ner/stanford-ner.jar"
export STANFORD_MODELS="$NLTK_DATA/stanford-ner/classifiers"

# download nltk trainer and prepare the NLTK taggers and chunkers used by seek
# warning: slow
git clone https://github.com/japerk/nltk-trainer
cd nltk-trainer
python setup.py install
python train_chunker.py treebank
python train_tagger.py brown

cd $cwd

# -- Gensim ---------------------------------------------------------------------
pip3 install -U numpy
pip3 install -U scipy
pip3 install -U regex
pip3 install -U gensim

# -- BLAS (basic linear algebra subroutines) & LAPACK (linear algebra)  ---------
if [[ $platform == 'linux' ]]; then
  sudo apt-get install gfortran
  sudo apt-get install liblapack-dev

  export LAPACK=/usr/lib/liblapack.so
  export ATLAS=/usr/lib/libatlas.so
  export BLAS=/usr/lib/libblas.so

elif [[ $platform == 'osx' ]]; then
  brew install gcc      # for fortran 

  curl http://www.netlib.org/blas/blas-3.6.0.tgz > blas.tgz
  tar xvf blas.tgz
  rm -f blas.tgz
  cd BLAS-3.6.0 
  make
  cp blas_LINUX.a libblas.a
  sudo cp libblas.a /usr/local/lib/ 
  cd ..
  rm -rf BLAS-3.6.0

  curl http://www.netlib.org/lapack/lapack-3.6.0.tgz > lapack.tgz
  tar xvf lapack.tgz
  rm -f lapack.tgz
  cd lapack-3.6.0
  cp make.inc.example make.inc
  make
  sudo cp liblapack.a /usr/local/lib/
  cd ..
  rm -rf lapack-3.6.0
  
fi


