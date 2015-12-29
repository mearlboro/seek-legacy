import numpy
import nltk
import os
import glob
import sys
from nltk import tokenize
from nltk.corpus import treebank


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



# -- linguist's commands -------------------------------------------------------------

# -- COMMAND vocab -------------------------------------------------------------------
# Get the union of the vocabularies of all documents in the source folder
def getvocab(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        words = getwords(filename)
        global vocab
        vocab = sorted(set(vocab + sorted(set([w.lower() for w in words])))) 
                                 # get vocabulary and add to total vocabulary
    print(vocab)
    return vocab 



# -- COMMAND freq --------------------------------------------------------------------
# Get the total word frequencies for all documents in the source folder
def getfrequency(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        words = getwords(filename)
        freq  = nltk.FreqDist([w.lower() for w in words])
        global freqs 
        freqs = freqs + freq     # find frequencies and add to total frequency distribution

    print(freqs)
    return freqs



# -- COMMAND freqsentences -----------------------------------------------------------
# Get the sentence frequency based on word frequency of all sentences in a document 
def filefreqsentences(filename):
    words = getwords(filename)
    sentences = getsents(filename)
    (lda_text, vocab, wordfreqs) = filedatokens(src)
    sentfreqs = []
    for sent in sentences:
        sentfreqs = sentfreqs + [(sent, sum(list(map(lambda word: word in vocab and wordfreqs.get(word) or 0, sent))))]

    return sorted(sentfreqs, key=lambda x:x[1], reverse=True)
    
# Get the sentence frequency based on word frequency of all sentences of all documents in the source folder
def getfreqsentences(src):
    # eliminate stop words, as they are not relevant when calculating the most relevant sentences
    (lda_text, vocab, wordfreqs) = getldatokens(src)
    for filename in glob.glob(os.path.join(src, '*.txt')):
        words = getwords(filename)
        sentences = getsents(filename) # 2D array of sentences 

        # when summing frequencies per sentence thus use wordfreqs
        global sentfreqs
        for sent in sentences:
            sentfreqs = sentfreqs + [(sent, sum(list(map(lambda word: word in vocab and wordfreqs.get(word) or 0, sent))))]

        # sort by relevance descending     
        sortedfreqs = sorted(sentfreqs, key=lambda x:x[1], reverse=True) 
        
        print(sortedfreqs)
        return sortedfreqs


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

# Remove the stop words and return the new text, vocabulary, and word frequence for a document
def fileldatokens(filename):
    text = getwords(filename)
    parts_of_speech = nltk.pos_tag(text)

    lda_parts_of_speech = filter(lambda  pos : pos[1] in filter_pos, parts_of_speech)
    lda_text = [pos[0].lower() for pos in lda_parts_of_speech]

    vocab = sorted(set(vocab + lda_text)) 
    freq  = nltk.FreqDist(lda_text)
    freqs = freqs + freq
    
    return (lda_text, vocab, freqs)

# Remove the stop words and return the new text, vocabulary, and word frequences for all documents in the source folder
def getldatokens(src):

    for filename in glob.glob(os.path.join(src, '*.txt')):
        text = getwords(filename)

        # use the tagger to identify and remove lda stop words
        # keep both vocabs and freqs
        parts_of_speech = nltk.pos_tag(text)

        # filter the good pos
        lda_parts_of_speech = filter(lambda  pos : pos[1] in filter_pos, parts_of_speech)
        lda_text = [pos[0].lower() for pos in lda_parts_of_speech]

        global vocab
        vocab = sorted(set(vocab + lda_text)) 
                                # get vocabulary and add to total vocabulary 
        
        freq  = nltk.FreqDist(lda_text)
        global freqs
        freqs = freqs + freq    # find frequencies and add to total frequency distribution
        
        print(lda_text)
        return (lda_text, vocab, freqs)



# COMMAND chunk 
# NP: nouns with prepositions, articles, and adjectives => entities with attributes
# VP: verbs (simple and compound), last verb in VP in infinitive =>  relations
# TODO: separate the verb to be! B.* tags no longer working <BEM>?<BER>?<BEZ>?<BEN>?<BED>?<BEDZ>?

grammar = '''
NP:   {<DT>?<JJ.*>*<NN.*>*}  
VP:   {<VBP>?<VBZ>?<VBD>?<RB>?<V.*>}
PREP: {<IN>}
PRON: {<PR.*>}
PP:   {<PREP>?<PRON>?<NP>}
OBJ:  {<IN><NP|PP>*} 
'''

def chunk(src):

    for filename in glob.glob(os.path.join(src, '*.txt')):
        sentences = getsents(filename)             # 2D array of sentences 
        parts_of_speech = list(map(nltk.pos_tag, sentences)) # 2D array of tuples (word, pos)
        chunker = Regexp(grammar)                       # will split words into groups as in grammar
        chunks  = map(chunker.parse, parts_of_speech)
        # for c in chunks: 
        #     c.draw() 
        

commands = {
    'vocab': getvocab,
    'freq': getfrequency,
    'freqsentences': getfreqsentences,
    'ldatokens': getldatokens,
    'chunk': chunk
}

if len(sys.argv) <= 2:
    print("the linguist expects the following command \n linguist.py <command> <src>")
    sys.exit(0)
if len(sys.argv) > 2:
    com = sys.argv[1]
    src = sys.argv[2]
    if not os.path.isdir(src): 
        print("<src> is not a directory")
        sys.exit(0)
    print("Executing linguist " + com + " on directory " + src + " ...")
    if commands.get(com, False):
        commands[com](src)
    else:
        print("<command> can be \n vocab \n freq \n freqsentences \n ldatokens \n chunk")
        sys.exit(0)    

