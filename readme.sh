# full install script for seek and all dependencies

## -- required: curl ----------------------------------------------------------
# sudo apt-get install curl

## -- ruby --------------------------------------------------------------------
#gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
#curl -sSL https://get.rvm.io | bash -s stable --ruby
#rvm install 2.1.5
#/bin/bash --login
#rvm use 2.1.5
#sudo gem install mechanize

## -- python ------------------------------------------------------------------
#sudo apt-get install python3
#sudo apt-get install python-virtualenv
#sudo apt-get install python3-pip

# -- setup environment --------------------------------------------------------
virtualenv -p /usr/bin/python3 seek-env
# -- dont forget to always activate / deactivate ------------------------------
source seek-env/bin/activate
cd seek-env

# -- installing textract for python 3 -----------------------------------------
# installer script based on http://www.tysonmaly.com/installing-textract-for-python-3/ 
apt-get install python3-dev libxml2 libxslt antiword unrtf poppler-utils pstotext tesseract-ocr flac lame libmad0 libsox-fmt-mp3 sox libjpeg-dev zlib1g-dev
pip3 install --upgrade lxml
curl https://pypi.python.org/packages/source/t/textract/textract-1.4.0.tar.gz > textract-1.4.0
tar xvf textract-1.4.0
cd textract-1.4.0
ed -s requirements/python <<< $',s/pdfminer==20140328/\nw' 
pip3 install pdfminer3k
python3 setup.py install
cd ..
ed -s lib/python3.4/site-packages/textract-1.4.0-py3.4.egg/textract/parsers/utils.py <<< $',s/(text, unicode)/(text, str)\nw'

