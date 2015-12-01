import numpy
import nltk
import textract
import os
import glob
import sys

# Use script by calling $ python linguist.py <source> <destination>

vocab = []
freqs = nltk.FreqDist('') 

# gets tokens and text in nltk text type
def gettext(src, filename):
    f = open(src + '/' + filename, 'r+')
    raw_text = f.read()
    toks  = nltk.word_tokenize(raw_text)    # tokenize raw text
    text  = nltk.Text(toks)               # nltk type text
    return (toks, text) 

def getvocab(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        (toks,text)  = gettext(src, filename)
        global vocab
        vocab = sorted(set(vocab + sorted(set([w.lower() for w in text])))) 
                                                # get vocabulary and add to total vocabulary 
    print(vocab)
 
def getfrequency(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        (toks,text)  = gettext(src, filename)
        freq  = nltk.FreqDist(text)
        global freqs 
        freqs = freqs + freq                    # find frequencies and add to total frequency distribution

    print(freqs)

def getldatokens(src):
    tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())  # this guy here imports a high-end English dictionary. warning: it's just slow, don't ctrl+c 
    for filename in glob.glob(os.path.join(src, '*.txt')):
        (toks,text)  = gettext(src, filename)

        # TODO: use the tagger to identify and remove lda stop words
        # keep both vocabs and freqs

    print(vocab)
    print(freqs)


commands = {
    'vocab': getvocab,
    'freq': getfrequency,
    'ldatokens': getldatokens,
}

if len(sys.argv) <= 2:
    print("the linguist expects the following command \n linguist.py <command> <src>")
    sys.exit(0)
if len(sys.argv) > 2:
    com = sys.argv[1]
    src = sys.argv[2]
    if commands.get(com, False):
        commands[com](src)
    else:
        print("<command> can be \n vocab \n freq \n ldatokens")
        sys.exit(0)    

