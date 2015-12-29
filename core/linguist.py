import numpy
import scipy
import nltk
import textract
import os
import glob
import sys
from gensim import corpora, models, similarities

# Use script by calling $ python linguist.py <command> <source>

vocab = []
freqs = nltk.FreqDist('')

# gets tokens and text in nltk text type
def gettext(src, filename):
    f = open(filename, 'r+')
    raw_text = f.read()
    toks  = nltk.word_tokenize(raw_text)    # tokenize raw text
    text  = nltk.Text(toks)                 # nltk type text
    return text


# COMMAND vocab
def getvocab(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        text  = gettext(src, filename)
        global vocab
        vocab = sorted(set(vocab + sorted(set([w.lower() for w in text]))))
                                 # get vocabulary and add to total vocabulary

    print(vocab)
    return vocab


# COMMAND freq
def getfrequency(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        text  = gettext(filename)
        freq  = nltk.FreqDist(text)
        global freqs
        freqs = freqs + freq     # find frequencies and add to total frequency distribution

    #print freqs
    return freqs


# COMMAND ldatokens
def getldatokens(src):
    # this guy here imports a high-end English dictionary. warning: it's just slow, don't ctrl+c
    tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

    # nltk.help.upenn_tagset() # to see all
    filter_pos = set([
         # 'CD'  , # numeral: cardinal
        'JJ'  ,  # ordinal adjective or numeral
        'JJR' ,  # comparative adjective
        'JJS' ,  # superlative adjective
        'NN'  ,  # singular or mass common noun
        'NNS' ,  # plural common noun
        'NNP' ,  # singular proper noun
        'NNPS',  # plural proper noun
        'RB'  ,  # adverb
        'RBR' ,  # comparative adverb
        'RBS' ,  # superlative adverb
        'VB'  ,  # verb
        'VBD' ,  # verb past tense
        'VBG' ,  # verb present participle or gerund
        'VBN' ,  # verb past participle
        'VBP' ,  # verb present
        'VBZ' ,  # verb present 3rd person singular
    ])

    for filename in glob.glob(os.path.join(src, '*.txt')):
        text = gettext(src, filename)

        # use the tagger to identify and remove lda stop words
        # keep both vocabs and freqs
        parts_of_speech = nltk.pos_tag(text)    # returns a list of tuples (tag, token)

        # filter the good pos
        lda_parts_of_speech = filter(lambda  pos : pos[1] in filter_pos, parts_of_speech)
        lda_text = [pos[0].lower() for pos in lda_parts_of_speech]

        global vocab
        vocab = sorted(set(vocab + lda_text))
                                # get vocabulary and add to total vocabulary

        freq  = nltk.FreqDist(lda_text)
        global freqs
        freqs = freqs + freq    # find frequencies and add to total frequency distribution

    #print vocab
    #print freqs
    return vocab, freqs


# Initial batch training.
def extracttopicsbatch(src):
    vocab, freqs = getldatokens(src)
    lda = gensim.models.ldamodel.LdaModel(corpus=vocab, id2word=freqs, num_topics=100, update_every=1, chunksize=500, passes=1)

    print(lda.print_topics(10))
# -- alternative printing
#    for i in range(0, lda.num_topics-1):
#        print(lda.print_topic(i))


# Updates.
def extracttopicsupdate(src):
    vocab, freqs = getldatokens(src)
    lsi = gensim.models.lsimodel.LsiModel(corpus=vocab, id2word=freqs, num_topics=100)

    print(lsi.print_topics(10))


commands = {
    'vocab': getvocab,
    'freq': getfrequency,
    'ldatokens': getldatokens,
    'topics': extracttopicsbatch,
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
