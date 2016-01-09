# Seek: linguist.py
# Interface to natural language processing and topic modelling

# ------------------------------------------------------------------------------------

'''
This script performs various NLP tasks by importing the trained classes in statistician.
The classes perform independent tasks and their functions are chained to perform:
    Topic Extraction
    Named Entity Extraction
    Text Summarisation
    Question Classification

Use script by calling $ python linguist.py <command> <source>

'''

# ------------------------------------------------------------------------------------

import os, glob, sys, pickle
import string
from operator  import *
from itertools import *
from functools import *

import numpy, nltk, gensim
from nltk.corpus import stopwords
from gensim import corpora, models, similarities


# ------------------------------------------------------------------------------------
''' import the trained classes from /skills '''
def getSentenceTokenizer():
    with open('skills/init_sent_tok.pkl','rb') as infile:
        st = pickle.load(infile)
    return st

def getChunkParser():
    with open('skills/init_chunk.pkl','rb') as infile:
        cp = pickle.load(infile)
    return cp

def getNameEntityDetector():
    with open('skills/init_ner.pkl','rb') as infile:
        ner = pickle.load(infile)
    return ner

def getTopicModelling():
    with open('skills/init_tm.pkl','rb') as infile:
        tm = pickle.load(infile)
    return tm

def getQuestionClassifier():
    with open('skills/init_qc.pkl','rb') as infile:
        qc = pickle.load(infile)
    return qc


''' Get text from document or directory '''
def getdocs(src):
    if os.path.isdir(src):  
        print("Collecting documents at directory " + src + " ...")
        documents = []
        for f in glob.glob(os.path.join(src, '*.txt')):
            documents += open(f, 'r+').read()
        return documents
    if os.path.isfile(src):
        print("Collecting document " + src + " ...")
        return [open(src, 'r+').read()]



######################################################################################
''' Helper functions for NLP '''

# Get the vocabulary of a document split in toks. 
def vocab(toks):
    voc = []
    voc = sorted(set(voc + sorted(set([w.lower() for w in toks]))))
    return voc


# Get the word frequencies of a set of tokens.
# The results of this function can be simply added for multiple texts
def word_freq(toks):
    freqs  = nltk.FreqDist([w.lower() for w in toks])
    return freqs


# Filter punctuation
def filter_punct(toks):
    words = list(filter(lambda w: w not in string.punctuation, toks)) 
    return words


# Filter out stop words and irrelevant parts of speech from a set of tokens
def filter_stop_words(toks):
    # List of parts of speech which are not stop words
    # nltk.help.upenn_tagset() to see all
    filter_pos = set([
        'CD'  ,  # numeral: cardinal
        'JJ'  ,  # ordinal adjective or numeral
        'JJR' ,  # comparative adjective
        'JJS' ,  # superlative adjective
        'NN'  ,  # singular or mass common noun
        'NNS' ,  # plural common noun
        'NNP' ,  # singular proper noun
        'NNPS',  # plural proper noun
        'RBR' ,  # comparative adverb
        'RBS' ,  # superlative adverb
        'VB'  ,  # verb
        'VBD' ,  # verb past tense
        'VBG' ,  # verb present participle or gerund
        'VBN' ,  # verb past participle
        'VBP' ,  # verb present
        'VBZ' ,  # verb present 3rd person singular
    ])
    # import the NLTK stopword corpus
    # words can be seen here http://snowball.tartarus.org/algorithms/english/stop.txt
    stopwords_corpus = stopwords.words('english')

    # use the tagger to identify part of speech
    parts_of_speech = nltk.pos_tag(toks)
    # filter out the pos of irrelevant words
    parts_of_speech_filter = filter(lambda  pair : pair[1] in filter_pos, parts_of_speech)
    # filter out the nltk stopwords corpus
    corpus_filter = filter(lambda pair : pair[0] not in stopwords_corpus, parts_of_speech_filter)

    # get the list of remaining tokens in original case and all lowercase
    filtered_text  = [pair[0] for pair in corpus_filter]
    filtered_lower = [word.lower() for word in filtered_text]

    return (filtered_text, filtered_lower)



# Get the weight of each sentence in a text based on frequency 
def sentence_freq(text):
    # get and filter words
    words = nltk.word_tokenize(text)
    words = filter_punct(words) 
    (filtered_words, filtered_lower) = filter_stop_words(words)
    
    # get vocab and freqs
    voc = vocab(filtered_lower)
    freqs = word_freq(filtered_lower)

    # Get the LSI topics of the text
    # TODO: replace with TopicModelling class?
    num = 10 # TODO: choose a number that has a relevance!!!
    dictionary = corpora.Dictionary([filtered_words])
    corp = [dictionary.doc2bow(filtered_words)]
    lsi_topics = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=dictionary, num_topics=num)
    # returns the topics as a dictionary of words and scores
    topics = topics2dict(lsi_topics.print_topics(num))

    # get sents
    sents = getSentenceTokenizer().text2sents(text)

    # when summing frequencies per sentence thus use wordfreqs
    # also add bias from topics in that phrase
    sentfreqs = []
    for sent in sents:
        sentfreqs += [(reduce(lambda x,y: x + ' ' + y, sent),
            numpy.mean(list(map(lambda word: word.lower() in voc and freqs.get(word) or 0, sent))) +
            sum(list(map(lambda word: word.lower() in topics.keys() and topics[word.lower()] or 0, sent)))
        )]

    # sort by relevance descending
    sortedfreqs = sorted(sentfreqs, key=lambda x:x[1], reverse=True)
    return sortedfreqs




#######################################################################################
# -- COMMAND summary ---------------------------------------------------------------------
def getsummary(src, args):
    return []


# -- COMMAND entities --------------------------------------------------------------------
def getentities(src, args):
    return []


# -- COMMAND topics ----------------------------------------------------------------------
# TODO: use the trained class

'''
Extracts topics by either LDA or LSI model, depending on args.
    <src> is a file or directory
    <args[0]> can be 0 (LDA) or 1 (LSI), default behaviour is LDA.
    <args[1]> must be an integer representing the number of topics to extract, default number is 10.
'''

def gettopics(src, args):
    if len(args) < 2:
        print("Incorrect arguments: expected \n linguist.py topics <src> <model> <num>")
        sys.exit(0)

    docs = getdocs(src)    

    # TODO: chunk mwes and collocations if necessary.
    # tokenize each doc, filter punctuation and stop words
    docs = list(map(filter_punct, map(nltk.word_tokenize, docs)))
    filtered = [f[1] for f in list(map(filter_stop_words, docs))]

    # create Gensim dictionary and corpus    
    dictionary = corpora.Dictionary(filtered) # choose the text with lowercase words
    corp = [dictionary.doc2bow(reduce(add, filtered))]

    if args[0] == 1:
        print("LSI model topics:")
        return lsi(dictionary, corp, args[1])
    else:
        print("LDA model topics:")
        return lda(dictionary, corp, args[1])


def topics2dict(topics):
    topics = topics[0][1].split('+')
    pairs  = [topic.split('*') for topic in topics]
    pairs  = [(''.join(list(filter(lambda c:c not in "\" ", pair[1]))), float(pair[0])) for pair in pairs]
    return dict(pairs)
    

def lsi(dictionary, corp, num):
    lsi_topics = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=dictionary, num_topics=num)

    # returns the topics as a dictionary of words and scores
    return topics2dict(lsi_topics.print_topics(num))


def lda(dictionary, corp, num):
    lda_topics = gensim.models.ldamodel.LdaModel(corpus=corp, id2word=dictionary, num_topics=num)
    return lda_topics.print_topics(num)


# -- COMMAND relationships ---------------------------------------------------------------
def getrelationships(src, args):
    return []




#######################################################################################
commands = {
    'summary': getsummary,
    'entities': getentities,
    'topics': gettopics,
    'relationships': getrelationships,
}

if len(sys.argv) <= 2:
    print("the linguist expects the following command \n linguist.py <command> <src>")
    sys.exit(0)
if len(sys.argv) > 2:
    com  = sys.argv[1]
    src  = sys.argv[2]
    args = 0
    if len(sys.argv) > 3:
        args = int(sys.argv[3]), int(sys.argv[4])

    print("Executing linguist " + com + " on " + src + " ...")
    if commands.get(com, False):
        print(commands[com](src, args))
    else:
        print("<command> can be \n summary \n entities \n topics \n relationships")
        sys.exit(0)

    print(sentence_freq(getdocs(src)[0]))
