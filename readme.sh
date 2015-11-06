# installer script based on http://www.tysonmaly.com/installing-textract-for-python-3/ 

## -- required: curl ----------------------------------------------------------
# sudo apt-get install curl

# -- setup environment --------------------------------------------------------
virtualenv -p /usr/bin/python3 seek-env
# -- dont forget to always activate / deactivate ------------------------------
source seek-env/bin/activate
cd seek-env

# -- installing textract for python 3 -----------------------------------------
apt-get install python-dev libxml2 libxslt antiword unrtf poppler-utils pstotext tesseract-ocr flac lame libmad0 libsox-fmt-mp3 sox
curl https://pypi.python.org/packages/source/t/textract/textract-1.4.0.tar.gz > textract-1.4.0
tar xvf textract-1.4.0
cd textract-1.4.0
ed -s requirements/python <<< $',s/pdfminer==20140328/\nw' 
pip install pdfminer3k
python3 setup.py install

