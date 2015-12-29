import numpy
import nltk
import os
import glob
import sys
from nltk import tokenize
from nltk.corpus import treebank


# Use script by calling $ python linguist.py <command> <source>

# -- globals and helpers ----------------------------------------------------
vocab = []
freqs = nltk.FreqDist('') 

# gets tokens and text in nltk text type
def gettext(src, filename):
    f = open(filename, 'r+')
    raw_text = f.read()
    toks  = nltk.word_tokenize(raw_text)    # tokenize raw text
    text  = nltk.Text(toks)                 # nltk type text
    return text 

# gets all sentences in the text as list of tokens and text in nltk text type
def getsents(src, filename):
    f = open(filename, 'r+')
    raw_text = f.read()
    sents = nltk.sent_tokenize(raw_text)    # tokenize raw text
    sents_toks = list(map(nltk.word_tokenize, sents))  
    text = list(map(nltk.Text, sents_toks))
    return text 



# -- linguist's commands ----------------------------------------------------

# COMMAND vocab
def getvocab(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        text  = gettext(src, filename)
        global vocab
        vocab = sorted(set(vocab + sorted(set([w.lower() for w in text])))) 
                                 # get vocabulary and add to total vocabulary
    print(vocab) 


# COMMAND freq
def getfrequency(src):
    for filename in glob.glob(os.path.join(src, '*.txt')):
        text  = gettext(src, filename)
        freq  = nltk.FreqDist([w.lower() for w in text])
        global freqs 
        freqs = freqs + freq     # find frequencies and add to total frequency distribution

    print(freqs)


# COMMAND ldatokens
def getldatokens(src):

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
    
    print(vocab)
    print(freqs)


# COMMAND chunk 
# NP: nouns with prepositions, articles, and adjectives => entities with attributes
# VP: verbs (simple and compound), last verb in VP in infinitive =>  relations
# TODO: separate the verb to be! B.* tags no longer working <BEM>?<BER>?<BEZ>?<BEN>?<BED>?<BEDZ>?

grammar = '''
NP:   {<DT>?<JJ.*>*<NN>*}  
VP:   {<VBP>?<VBZ>?<VBD>?<RB>?<V.*>}
PREP: {<IN>}
PRON: {<PR.*>}
PP:   {<PREP>?<PRON>?<NP>}
OBJ:  {<IN><NP|PP>*} 
'''

def chunk(src):
    # this guy here imports a high-end English dictionary. warning: it's just slow, don't ctrl+c 
    tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

    for filename in glob.glob(os.path.join(src, '*.txt')):
        sentences = getsents(src, filename) # 2D array of sentences 
        parts_of_speech = list(map(nltk.pos_tag, text)) # 2D array of tuples (word, pos)
        chunker = Regexp(grammar)                       # will split words into groups as in grammar
        chunks  = map(chunker.parse, parts_of_speech)
        # for c in chunks: 
        #     c.draw() 
        

commands = {
    'vocab': getvocab,
    'freq': getfrequency,
    'ldatokens': getldatokens,
    'chunk': chunk
}

if len(sys.argv) <= 2:
    print("the linguist expects the following command \n linguist.py <command> <src>")
    sys.exit(0)
if len(sys.argv) > 2:
    com = sys.argv[1]
    src = sys.argv[2]
    print("Executing linguist " + com + " on directory " + src + " ...")
    if commands.get(com, False):
        commands[com](src)
    else:
        print("<command> can be \n vocab \n freq \n ldatokens \n chunk")
        sys.exit(0)    

