#!/bin/bash

platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
  platform='osx'
fi


## -- ruby --------------------------------------------------------------------
# gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
curl -sSL https://get.rvm.io | bash -s stable --ruby
rvm install 2.2.1
/bin/bash --login
rvm use 2.2.1
sudo gem install mechanize

if [[ $platform == 'linux' ]]; then
  # -- required: curl ----------------------------------------------------------
  sudo apt-get install curl

    ## -- python ------------------------------------------------------------------
  sudo apt-get install python3
  sudo apt-get install python-virtualenv
  sudo apt-get install python3-pip

  # -- setup environment --------------------------------------------------------
  virtualenv -p /usr/bin/python3 seek-env
  # -- dont forget to always activate / deactivate ------------------------------
  /bin/bash -c ". seek-env/bin/activate; exec /bin/bash -i"
  cd seek-env

  # -- installing textract for python 3 -----------------------------------------
  # installer script based on http://www.tysonmaly.com/installing-textract-for-python-3/
  apt-get install python3-dev libxml2 antiword unrtf poppler-utils pstotext tesseract-ocr flac lame libmad0 libsox-fmt-mp3 sox libjpeg-dev zlib1g-dev
  pip3 install --upgrade lxml
  curl https://pypi.python.org/packages/source/t/textract/textract-1.4.0.tar.gz > textract-1.4.0
  tar xvf textract-1.4.0
  cd textract-1.4.0
  ed -s requirements/python <<< $',s/pdfminer==20140328/\nw'
  pip3 install pdfminer3k
  python3 setup.py install
  cd ..
  ed -s lib/python3.4/site-packages/textract-1.4.0-py3.4.egg/textract/parsers/utils.py <<< $',s/(text, unicode)/(text, str)\nw'

  # -- installng gensim for topic modelling -------------------------------------
  pip3 install -U gensim

  # -- installing NLTK for python 3 and its dependencies ------------------------
  # nltk.org
  pip3 install -U nltk
  pip3 install -U numpy
  python -m nltk.downloader all
elif [[ $platform == 'osx' ]]; then

  brew install python3

  # -- setup environment --------------------------------------------------------
  virtualenv -p /usr/local/bin/python3 seek-env
  # -- dont forget to always activate / deactivate ------------------------------
  /bin/bash -c ". seek-env/bin/activate; exec /bin/bash -i"
  cd seek-env

  brew install libxml2 antiword unrtf poppler tesseract flac lame libmad libsoxr libjpeg zlib
  pip3 install --upgrade lxml

  curl https://pypi.python.org/packages/source/t/textract/textract-1.4.0.tar.gz > textract-1.4.0.tar.gz
  tar xvf textract-1.4.0.tar.gz
  cd textract-1.4.0
  ed -s requirements/python <<< $',s/pdfminer==20140328/\nw'
  pip3 install pdfminer3k
  python3 setup.py install
  cd ..
  ed -s lib/python3.5/site-packages/textract-1.4.0-py3.5.egg/textract/parsers/utils.py <<< $',s/(text, unicode)/(text, str)\nw'
  pip3 install -U gensim
  pip3 install -U nltk
  pip3 install -U numpy
  python3 -m nltk.downloader all
fi
