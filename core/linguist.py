import numpy
import nltk
import gensim
import os
import glob
import sys
import numpy
from itertools import *
from functools import *
from nltk.corpus import stopwords
from gensim import corpora, models, similarities

# Use script by calling $ python linguist.py <command> <source>

# -- globals and helpers -------------------------------------------------------------
vocab = []
freqs = nltk.FreqDist('')
sentfreqs = []

# gets tokens and text in nltk text type
def getwords(filename):
    f = open(filename, 'r+')
    raw_text = f.read()
    words = nltk.word_tokenize(raw_text)    # tokenize raw text
    text  = nltk.Text(words)                # nltk type text
    return text

# gets all sentences in the text as list of tokens and text in nltk text type
def getsents(filename):
    f = open(filename, 'r+')
    raw_text = f.read()
    sents = nltk.sent_tokenize(raw_text)    # tokenize into sentences
    sents_words = list(map(nltk.word_tokenize, sents)) # tokenize each sentence into words
    text = list(map(nltk.Text, sents_words))
    return text



# -- analysis ------------------------------------------------------------------------

# -- COMMAND vocab -------------------------------------------------------------------
# Get the union of the vocabularies of all documents in the source folder
def getvocab(src, args):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        words = getwords(filename)
        global vocab
        vocab = sorted(set(vocab + sorted(set([w.lower() for w in words]))))
                                 # get vocabulary and add to total vocabulary
    return vocab



# -- COMMAND freq --------------------------------------------------------------------
# Get the total word frequencies for all documents in the source folder
def getfrequency(src, args):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        words = getwords(filename)
        freq  = nltk.FreqDist([w.lower() for w in words])
        global freqs
        freqs = freqs + freq     # find frequencies and add to total frequency distribution

    return freqs



# -- natural language processing -----------------------------------------------------

# -- COMMAND chunk -------------------------------------------------------------------
# COMMAND chunk
grammar = '''
NP:   {<DT>?<JJ.*>*<NN.*>*}
VP:   {<VBP>?<VBZ>?<VBD>?<RB>?<V.*>}
PREP: {<IN>}
PRON: {<PR.*>}
PP:   {<PREP>?<PRON>?<NP>}
OBJ:  {<IN><NP|PP>*}
'''

def chunk(src, args):

    for filename in glob.glob(os.path.join(src, '*.txt')):
        sentences = getsents(filename)                       # 2D array of sentences
        parts_of_speech = list(map(nltk.pos_tag, sentences)) # 2D array of tuples (word, pos)
        chunker = nltk.RegexpParser(grammar)                 # will split words into groups as in grammar
        chunks  = list(map(chunker.parse, parts_of_speech))
        # for c in chunks:
            # c.draw()

        return chunks



# -- COMMAND ldatokens ---------------------------------------------------------------
# List of parts of speech which are not stop words
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
    # 'RB'  ,  # adverb
    'RBR' ,  # comparative adverb
    'RBS' ,  # superlative adverb
    'VB'  ,  # verb
    'VBD' ,  # verb past tense
    'VBG' ,  # verb present participle or gerund
    'VBN' ,  # verb past participle
    'VBP' ,  # verb present
    'VBZ' ,  # verb present 3rd person singular
])
# importing the NLTK stopword corpus
# words can be seen here http://snowball.tartarus.org/algorithms/english/stop.txt
stopwords_corpus = stopwords.words('english')

# Helper to filter out stop words
def filter_stop_words(text):
    # use the tagger to identify part of speech
    parts_of_speech = nltk.pos_tag(text)
    # filter out the pos of stop words
    lda_parts_of_speech_filter = filter(lambda  pair : pair[1] in filter_pos, parts_of_speech)
    # get the text in regular and all lowercase
    # filter out the nltk stopwords corpus
    lda_corpus_filter = filter(lambda pair : pair[0] not in stopwords_corpus, lda_parts_of_speech_filter)
    lda_text  = [pair[0] for pair in lda_corpus_filter]
    lda_lower = [word.lower() for word in lda_text]

    return (lda_text, lda_lower)


# Remove the stop words and return the new text, vocabulary, and word frequence for a document
def fileldatokens(filename):

    text = [w.lower() for w in getwords(filename)]

    (lda_text, lda_lower) = filter_stop_words(text)

    vocab = sorted(set(vocab + lda_lower))
    freq  = nltk.FreqDist(lda_lower)
    freqs = freqs + freq

    return (lda_text, vocab, freqs)

# Remove the stop words and return the new text, vocabulary, and word frequences for all documents in the source folder
def getldatokens(src, args):

    lda_texts = []
    for filename in glob.glob(os.path.join(src, '*.txt')):
        text = [w.lower() for w in getwords(filename)]

        (lda_text, lda_lower) = filter_stop_words(text)

        global vocab
        vocab = sorted(set(vocab + lda_lower))
                                # get vocabulary and add to total vocabulary

        freq  = nltk.FreqDist(lda_lower)
        global freqs
        freqs = freqs + freq    # find frequencies and add to total frequency distribution
        
        lda_texts += lda_text

    return (lda_texts, vocab, freqs)



# -- COMMAND freqsentences -----------------------------------------------------------
# Get the sentence frequency based on word frequency of all sentences in a document
def filefreqsentences(filename):
    words = getwords(filename)
    sentences = getsents(filename)
    (lda_text, vocab, wordfreqs) = filedatokens(src)
    topics = gettopicsupdate(src,0,8)
    sentfreqs = []
    for sent in sentences:
        sentfreqs = sentfreqs + [(sent, numpy.mean(list(map(lambda word: word in vocab and wordfreqs.get(word) or 0, sent))) + 0)]

    return sorted(sentfreqs, key=lambda x:x[1], reverse=True)

# Get the sentence frequency based on word frequency of all sentences of all documents in the source folder
# <args> represents number of relevant sentences returned, all if 0
def getfreqsentences(src, args):
    # eliminate stop words, as they are not relevant when calculating the most relevant sentences
    (lda_text, vocab, wordfreqs) = getldatokens(src, args)
    topics = extracttopicsupdate(src,[0,8])
    print(topics) 
    for filename in glob.glob(os.path.join(src, '*.txt')):
        words = getwords(filename)
        sentences = getsents(filename) # 2D array of sentences

        # when summing frequencies per sentence thus use wordfreqs
        # also add bias from topics in that phrase
        global sentfreqs
        for sent in sentences:
            sentfreqs += [(reduce(lambda x,y: x + ' ' + y, sent),
                numpy.mean(list(map(lambda word: word.lower() in vocab and wordfreqs.get(word) or 0, sent))) +
                sum(list(map(lambda word: word.lower() in topics.keys() and topics[word.lower()] or 0, sent)))  
            )]

        # sort by relevance descending
        sortedfreqs = sorted(sentfreqs, key=lambda x:x[1], reverse=True)

        if args == 0:
            return sortedfreqs
        else:
            return list(islice(sortedfreqs, 10))



# -- COMMAND topics ----------------------------------------------------------------------
# Extract topics.
# <src> must be a directory, <args[0]> can be 0 (initial) or 1 (update), default behaviour is initial.
# <args[1]> must be an integer representing the number of topics to extract, default number is 10.
def gettopics(src, args):
    if args[0] == 1:
        print("Update")
        return extracttopicsupdate(src, args)
    else:
        print("Initial")
        return extracttopicsinitial(src, args)

def extracttopicsupdate(src, args):
    (lda_text, vocab, freqs) = getldatokens(src, args)
    num = args[1]

    dictionary = corpora.Dictionary([lda_text])
    corp = [dictionary.doc2bow(lda_text)]

    lsi_topics = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=dictionary, num_topics=num)

    # returns the topics as a dictionary of words and scores
    topics = lsi_topics.print_topics(num)[0][1].split('+')
    pairs = [topic.split('*') for topic in topics]
    pairs = [(''.join(list(filter(lambda c:c not in "\" ", pair[1]))), float(pair[0])) for pair in pairs]
    return dict(pairs)


def extracttopicsinitial(src, args):
    (lda_text, vocab, freqs) = getldatokens(src, args)
    num = args[1]

    dictionary = corpora.Dictionary([lda_text])
    corp = [dictionary.doc2bow(lda_text)]

    lda_topics = gensim.models.ldamodel.LdaModel(corpus=corp, id2word=dictionary, num_topics=num)

    print(lda_topics.print_topics(num))
    return lda_topics



# -----------------------------------------------------------------------------------
commands = {
    'vocab': getvocab,
    'freq': getfrequency,
    'freqsentences': getfreqsentences,
    'chunk': chunk,
    'ldatokens': getldatokens,
    'topics': gettopics,
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
    if not os.path.isdir(src):
        print("<src> is not a directory")
        sys.exit(0)
    print("Executing linguist " + com + " on directory " + src + " ...")
    if commands.get(com, False):
        print(commands[com](src, args))
    else:
        print("<command> can be \n vocab \n freq \n freqsentences \n ldatokens \n chunk")
        sys.exit(0)
