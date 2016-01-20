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
from optparse import OptionParser

import numpy, nltk, gensim
from nltk.corpus import stopwords
from gensim import corpora, models, similarities

import logging
logger = logging.getLogger('handler')

from nltk.chunk.util import *
from nltk.chunk import *
from nltk.chunk.regexp import *
from nltk import nonterminals, Production, CFG

# ------------------------------------------------------------------------------------
''' Import the trained classes from /skills '''
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


''' Get text from document or directory. '''
def getdocs(src, pretty):
    if os.path.isdir(src):
        if not pretty:
            print("Collecting documents at directory " + src + " ...")

        documents = []
        for f in glob.glob(os.path.join(src, '*.txt')):
            documents += [open(f, 'r+').read()]
        return documents
    if os.path.isfile(src):
        if not pretty:
            print("Collecting document " + src + " ...")
        return [open(src, 'r+').read()]


######################################################################################
''' Helper functions for NLP '''

# Get the vocabulary of a document split in tokens.
def vocab(toks):
    voc = []
    voc = sorted(set(voc + sorted(set([w.lower() for w in toks]))))
    return voc


# Get the word frequencies of a set of tokens.
# The results of this function can be simply added for multiple texts.
def word_freq(toks):
    freqs  = nltk.FreqDist([w.lower() for w in toks])
    return freqs


# Filter punctuation.
def filter_punct(toks):
    words = list(filter(lambda w: w not in string.punctuation, toks))
    return words


# Filter out stop words and irrelevant parts of speech from a set of tokens.
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
    # filter out the parts of speech of irrelevant words
    parts_of_speech_filter = filter(lambda  pair : pair[1] in filter_pos, parts_of_speech)
    # filter out the nltk stopwords corpus
    corpus_filter = filter(lambda pair : pair[0] not in stopwords_corpus, parts_of_speech_filter)

    # get the list of remaining tokens in original case and all lowercase
    filtered_text  = [pair[0] for pair in corpus_filter]
    filtered_lower = [word.lower() for word in filtered_text]

    return (filtered_text, filtered_lower)



# Get the weight of each sentence in a text based on frequency.
def sentence_freq(text, sents):
    # get and filter words
    words = nltk.word_tokenize(text)
    words = filter_punct(words)
    (filtered_words, filtered_lower) = filter_stop_words(words)

    # get vocab and freqs
    voc = vocab(filtered_lower)
    freqs = word_freq(filtered_lower)

    # when summing frequencies per sentence thus use wordfreqs
    sentfreqs = []
    for sent in sents:
        sentfreqs += [(sent, numpy.mean(list(map(lambda word: word.lower() in voc and freqs.get(word) or 0, sent))))]

    return sentfreqs



###########################################################################################

# -- COMMAND mostfreq ---------------------------------------------------------------------
def getmostfreq(src, args, pretty):

    if not pretty:
        print("The most frequent word is ...")

    docs = getdocs(src, pretty)
    count = 0
    words = []
    freqs = []
    sortedfreqs = []
    for doc in docs:
        words.append(nltk.word_tokenize(doc))
        freqs.append(word_freq(words[count]))
        sortedfreqs.append(sorted(freqs[count].items(), key=lambda x:x[1], reverse=True))
        count += 1

    if not pretty:
        return sortedfreqs 
    else:
        return prettifymostfreq(sortedfreqs)



# -- COMMAND summary ---------------------------------------------------------------------
'''
When summing frequencies per sentence add bias from topics in that phrase
    model: 0 for LDA, 1 for LSI
    text : the content of a document in a string
    sents: the sentences in a document - a list of lists of word and punctuation tokens
    freqs: the output of sentence_frequency(text, sents)
'''
def augment_topics(model, text, sents, freqs):
    num = 10

    if model == 0:
        topics = lda2dict(lda([text], num))[1]
    else:
        topics = lsi2dict(lsi([text], num))

    sentfreqs = []
    for sent,freq in freqs:
        sentfreqs +=  [(reduce(lambda x,y: x + ' ' + y, sent),
                        freq + sum(list(map(lambda word: word.lower() in topics.keys() and topics[word.lower()] or 0, sent)))
                      )]
    return sentfreqs

'''
When summing frequencies per sentence add bias from named entities in that phrase
    text : the content of a document in a string
    sents: the sentences in a document - a list of lists of word and punctuation tokens
    freqs: the output of sentence_frequency(text, sents)
'''
def augment_ne(text, sents, freqs):
    bias = 2

    nes = ne([text], 1) # get single-word NEs by means of text analysis
    nes = [ne[0] for ne in nes] # the words as list

    sentfreqs = []
    for sent,freq in freqs:
        sentfreqs +=  [(reduce(lambda x,y: x + ' ' + y, sent),
                        freq * sum(list(map(lambda word: word.lower() in nes and bias or 1, sent)))
                      )]
    return sentfreqs


'''
Obtains a summary of each text in a directory or the text in a file, by choosing the
sentences of the highest augmented frequency:
The augmented frequency is calculated as the average word frequencies of the filtered
words in each sentence (as returned by sentence_freq), summed with a bias coming from
the presence of topics, named entities, or both in the sentence.

    <src> is a file or directory
    <args[0]> can be 0 (LDA), 1 (LSI), 2 (NEs), default behaviour is LDA.
    <args[1]> must be an integer representing the number of topics to extract, default number is 10.
'''
def getsummary(option, opt_str, value, parser):
    args = parser.rargs
    if len(args) < 4:
        print("Incorrect arguments: expected \n linguist.py summary <src> <model> <num>")
        sys.exit(0)
    src = args[0]
    model = args[1]
    num = int(args[2])
    pretty = args[3]

    text = "Constructing summary for documents at {} ..."
    print(text.format(src))

    docs = getdocs(src, pretty)

    st = getSentenceTokenizer()

    summaries = []
    for doc in docs:
        sents = st.text2sents(doc)
        freqs = sentence_freq(doc, sents)
        if model == 0 or model == 1:
            freqs = augment_topics(model, doc, sents, freqs)
        elif model == 2:
            freqs = augment_ne(doc, sents, freqs)
        sortedfreqs = sorted(freqs, key=lambda x:x[1], reverse=True)
        min_freq = sortedfreqs[num][1]
        summary = [f[0] for f in list(filter(lambda f: f[1] >= min_freq, freqs))]

        summaries += [summary]

    del st

    if not pretty:
      setattr(parser.values, option.dest, summaries)
    else:
      setattr(parser.values, option.dest, prettifysummary(summaries))

# -- COMMAND entities --------------------------------------------------------------------
'''
Finds the named entities after tagging and chunking the sentence with the trained Name
Entity Detector. For each document it returns a triple consisting of the regular named
entities found with the classifier, and focused and relevant name entities found with
sentence frequency measurements.

    <src> is a file or directory
    <args[0]> can be 0 (chunk-based detection) or 1 (text-based detection).
    <args[1]> can be 0 (NEs)
'''

def ne(docs, model):
    st  = getSentenceTokenizer()
    ch  = getChunkParser()
    ner = getNameEntityDetector()

    for doc in docs:
        sents  = st.text2sents(doc)
        if model == 0:
            chunks = ch.sents2chunks(sents)
            nes    = ner.chunks2ne(doc, chunks)
        else:
            nes    = ner.text2ne(doc)
        entities += [nes]

    del st
    del ch
    del ner

    return entities

def getentities(option, opt_str, value, parser):
    args = parser.rargs
    if len(args) < 4:
        print("Incorrect arguments: expected \n linguist.py --entities <src> <model>")
        sys.exit(0)
    src = args[0]
    model = args[1]
    model_name = 'chunk-based' if model else 'text-based'

    ntype = args[2]
    pretty = args[3]
    if ntype < 0 or ntype > 3:
        print("Incorrect arguments: <type> is 0 (PERSON), 1 (LOCATION), 2 (TIME), 3 (ORGANIZATION)")
    nes_dict = { 0: 'PERSON', 1: 'LOCATION', 2: 'TIME', 3: 'ORGANIZATION' }
    type_name = nes_dict[ntype].lower()

    if not pretty:
        out_text = "Retrieving {} named entities by {} detection for documents at ..."
        print(text.format(type_name, model_name, src))

    docs  = getdocs(src, pretty)
    nes = ne(docs, model)
    selected_nes = [list(filter(lambda n: n[1] == nes_dict[ntype], nesdoc)) for nesdoc in nes]

    if not pretty:
        setattr(parser.values, option.dest, selected_nes)
    else:
        setattr(parser.values, option.dest, prettifyentities(selected_nes, type_name))


# -- COMMAND topics ----------------------------------------------------------------------

'''
Converts output of lsi or lda to a dictionary of words and their weight
'''

def gettopics(option, opt_str, value, parser):
    args = parser.rargs
    if len(args) < 3:
        print("Incorrect arguments: expected \n linguist.py topics <src> <model> <pretty> <num>")
        sys.exit(0)
    src = args[0]
    model = args[1]
    model_name = 'LDA' if not model else 'LSI'
    pretty = args[2]
    num = args[3]
    if not pretty:
        print("Retrieving topics by the " + model_name + " model for documents at " + src + " ...")

    docs = getdocs(src, pretty)
    topics = []

    for doc in docs:
        if model == 1:
            topics.append(lsi2dict(lsi(docs, args[3])))
        else:
            topics.append(lda2dict(lda(docs, args[3]))[0])

    if not pretty:
        setattr(parser.values, option.dest, topics)
    else:
        setattr(parser.values, option.dest, prettifytopics(topics))

def lsi2dict(topics):
    topics = topics[0][1].split('+')
    pairs  = [topic.split('*') for topic in topics]
    pairs  = [(''.join(list(filter(lambda c:c not in "\" ", pair[1]))), float(pair[0])) for pair in pairs]
    return dict(pairs)

def lda2dict(topics):
    dicts = []
    for i in range(len(topics)):
        topic = topics[i][1].split('+')
        pairs = [t.split('*') for t in topic]
        pairs = [(''.join(list(filter(lambda c:c not in " ", pair[1]))), float(pair[0])) for pair in pairs]
        dicts += [dict(pairs)]

    return dicts


'''
Get lsi and lda topics 
'''
def lsi(docs, num):
    # tokenize each doc, filter punctuation and stop words
    docs = list(map(filter_punct, map(nltk.word_tokenize, docs)))
    filtered = [f[1] for f in list(map(filter_stop_words, docs))]

    # create Gensim dictionary and corpus
    dictionary = corpora.Dictionary(filtered) # choose the text with lowercase words
    corp = [dictionary.doc2bow(reduce(add, filtered))]

    lsi_topics = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=dictionary, num_topics=num)

    # returns the topics as a dictionary of words and scores
    return lsi_topics.print_topics(num)


def lda(docs, num):
    # tokenize each doc, filter punctuation and stop words
    docs = list(map(filter_punct, map(nltk.word_tokenize, docs)))
    filtered = [f[1] for f in list(map(filter_stop_words, docs))]

    # create Gensim dictionary and corpus
    dictionary = corpora.Dictionary(filtered) # choose the text with lowercase words
    corp = [dictionary.doc2bow(reduce(add, filtered))]

    lda_topics = gensim.models.ldamodel.LdaModel(corpus=corp, id2word=dictionary, num_topics=num)
    return lda_topics.print_topics(num)

# -- COMMAND relationships ---------------------------------------------------------------
'''
Below is the information extractor. It chooses the sentences containing topics or named
entities, then chunks them to extact the correct parts of speech, analyse them, and pair
them into sets of attributes or relationships
    <src> can be a file or directory
    takes no args
'''

'''
Turns a phrase like 'Hawaii is warm' into the dictionary item
'Hawaii': { 'LOCATION', 'warm' }
If in any other sentence, we find 'volcanoes erupt in the hot island of Hawaii', then
'Hawaii': { 'LOCATION', 'warm', 'hot island' }

Makes use of splitting the sentence into chunks and looks for NPs, VPs

NP: nouns with prepositions, articles, and adjectives => entities with attributes
VP: verbs (simple and compound), last verb in VP in infinitive =>  relations
'''
def node_children(tree):
    return [t  for t in list(islice(tree.subtrees(), len(tree)))]

def parse_NP(tree):
    return []


def relations(sents, chunks, nes, ldas):

    adjs = [ 'JJ', 'JJR', 'JJS' ]
    vbs  = [ 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' ]
    nns  = [ 'NN', 'NNS', 'NNP', 'NNPS' ]
    ins  = [ 'of', 'that', 'which', 'like', 'in', 'at', 'as' ]

    nes_merged = dict([(' '.join(n[0]), n[1]) for n in nes])
    pers_org = dict(filter(lambda t: t[1] == 'PERSON', nes_merged.items()))
    # dictionary of dictionaries for each named entity
    retrieved = {}
    # Previously found named entity and the most likely candidate for the attributes
    prev_ne = None
    relations = {}
    relation = []
    if len(chunks) > 1:
        for chunked_sent in chunks:
            # filter through all chunks for NPs, get each subtree that may contain attributes next to nouns
            # then create dictionary entry for the named entity/noun if it does not exist
            # and add the adjectives in the bag
            filtered_chunked_subtrees = list(chunked_sent.subtrees(filter= lambda t: t.label() == 'NP' or t.label() == 'CD'))
            # merge noun that comes after noun phrase into a noun phrase
            # for subtree in filtered_chunked_subtrees:
            for subtree in chunked_sent.subtrees():
                if subtree in filtered_chunked_subtrees:
                    # filter attributes and NEs
                    sentence = [t[0] for t in subtree.leaves() if t[1] != 'PRP' and t[1] not in vbs]
                    ent_key = []
                    atr = []
                    for key in pers_org.keys():
                        for word in sentence:
                            if word in key.split():
                                ent_key.append(word)
                            else:
                                atr.append(word)
                    ent_key = ' '.join(ent_key)
                    atr = ' '.join(atr)
                    if ent_key != "":
                        prev_ne = ent_key
                    if prev_ne not in retrieved.keys():
                        if atr != "":
                            retrieved[prev_ne] = [atr]
                    else:
                        if atr != "":
                            retrieved[prev_ne].append(atr)
                else:
                    # Analyze sentence structure, extract and map verbs to right
                    # NEs
                    S, NP, VP, PP = nonterminals('S, NP, VP, PP')
                    N, V, P, Det = nonterminals('N, V, P, Det')
                    prods = subtree.productions()[0].rhs()
                    for prod in prods:
                        if prod == NP:
                            if prev_ne != None:
                                if prev_ne not in relations.keys():
                                    if len(relation) > 0:
                                        relations[prev_ne] = relation
                                else:
                                    if len(relation) > 0:
                                        relations[prev_ne].append(' '.join(relation).strip())
                                relation = []
                        else:
                            if prod[1] in vbs:
                                relation.append(prod[0])
                            elif prod[1] == 'IN':
                                relation.append("")
    elif len(chunks) == 1:
        chunked_sent = chunks[0]
        filtered_chunked_subtrees = list(chunked_sent.subtrees(filter= lambda t: t.label() == 'NP' or t.label() == 'CD'))
        # merge noun that comes after noun phrase into a noun phrase
        # for subtree in filtered_chunked_subtrees:
        for subtree in chunked_sent.subtrees():
            relation.extend([t[0] for t in subtree.leaves() if t[1] in vbs])
            if subtree in filtered_chunked_subtrees:
                # Select NE and attributes for single sentence
                ent_key = ' '.join([t[0] for t in subtree.leaves() if t[1] != 'PRP' and t[1] not in vbs])
                if ent_key != "":
                    if any(word in ent_key for word in pers_org.keys()):
                        prev_ne = ent_key
                    elif prev_ne not in retrieved.keys():
                        retrieved[prev_ne] = [ent_key]
                    else:
                        retrieved[prev_ne].append(ent_key)
                if prev_ne != None:
                    if prev_ne not in relations.keys():
                        if len(relation) > 0:
                            relations[prev_ne] = relation
                    else:
                        if len(relation) > 0:
                            relations[prev_ne].append(' '.join(relation).strip())
                    relation = []
    return retrieved, relations

'''
Barack Obama is the prezident of the United States.

NP Barack Obama
VP is
NP the prezident of the United States

VP = is, was, etc.

is(barack, prezinf)

person(barack, prezindent of the united states, born in hawaii
'''

def getrelationships(option, opt_str, value, parser):
    args = parser.rargs
    src = args[0]
    args = args[1:]
    print("Extracting information from documents at " + src + " ...")

    docs = getdocs(src)

    st  = getSentenceTokenizer()
    ch  = getChunkParser()
    ner = getNameEntityDetector()

    dbs = []

    for doc in docs:
        db = {}
        # Construct a dictionary of the form: Value of NE, relation, [(attribute, NE tag)]
        sents  = st.text2sents(doc)
        sents  = [list(filter(lambda x: x not in string.punctuation, sent)) for sent in sents]
        ldas   = [l for l in lda2dict(lda([doc], 2))[0]]
        regular_nes = list(ner.text2ne(doc))
        date_nes = dict(filter(lambda t: t[1] == "DATE", regular_nes))
        nes    = [ne[0] for ne in regular_nes]
        sents  = list(filter(lambda sent: any([t in sent for t in ldas]) or any([ne in sent for ne in nes]), sents))
        chunks = ch.sents2chunks(sents)
        nes    = ner.chunks2ne(doc, chunks)
        ats, rels = relations(sents, chunks, nes, ldas)
        nes = dict([(' '.join(n[0]), n[1]) for n in nes])
        for ent in ats.keys():
            prev_rel = None
            index = 1
            if ent in rels.keys():
                for relation in rels[ent]:
                    attributes = []
                    index += 1
                    if relation != "":
                        prev_rel = relation
                        for atrb in ats[ent][0:index]:
                            if atrb in nes.keys():
                                attributes.append((atrb, nes[atrb]))
                            elif ent in nes.keys():
                                attributes.append((atrb, nes[ent]))
                        if ent in nes.keys():
                            dbs += [((ent, nes[ent]), prev_rel, attributes)]
                        else:
                            words = ent.split()
                            for word in words:
                                if word in nes.keys():
                                    dbs += [((ent, nes[word]), prev_rel, attributes)]
                        attributes = []
                        del ats[ent][0:index]
                        index = 1
    setattr(parser.values, option.dest, dbs)

# -- COMMAND questions ------------------------------------------------------------------
'''
Classifies questions based on the question classifier
    <text> is the question in string format
    takes no args
'''
def getquestiontype(option, opt_str, value, parser):
    text = parser.rargs[0]
    qc  = getQuestionClassifier()
    ner = getNameEntityDetector()

    nes = ner.text2ne(text)
    c = qc.classify(text, nes)

    del qc
    del ner
    setattr(parser.values, option.dest, c)



##########################################################################################
''' 
These functions format the output for the 'conversational' web interface
'''

def prettifymostfreq(sortedfreqs):
    text =  'The most frequent word in this text is "{}" appearing {} times.'

    return text.format(sortedfreqs[0][0][0], sortedfreqs[0][0][1])

def prettifysummary(summaries):
    return [ ' '.join(summary) for summary in summaries][0]

def prettifyentities(nes, type_name):
    just_nes = [[ne[0] for ne in docne] for docne in nes][0]

    text0 = "I'm sorry, I don't think there is any " + type_name + " in your document." 
    text1 = "The " + type_name + " in your document is {}."
    textn = "The documents contain information about {} and {}."

    if len(just_nes) == 0:
        text = text0
    elif len(just_nes) == 1:
        text = text1.format(just_nes[0])
    else:
        text = textn.format(', '.join(just_nes[:-1]), just_nes[-1])

    return text

def prettifytopics(topics):
    just_topics = sorted(set([t[0] for t in topics.items()]), reverse=True)
    text = 'The document you gave me to read is about "{}" and also mentions "{}" and "{}" rather insistently.'

    return text.format(just_topics[0], just_topics[1], just_topics[2])

def prettifyrelationships(relationships):
    return "TODO:"


##########################################################################################
parser = OptionParser()
parser.add_option("-e", "--entities", help="Extract named entities", action="callback", callback=getentities, dest="output")
parser.add_option("-s", "--summary", help="Offer summary of text", action="callback", callback=getsummary, dest="output")
parser.add_option("-t", "--topics", help="Extract topics from text", action="callback", callback=gettopics, dest="output")
parser.add_option("-r", "--relationships", help="Extract relationships from text", action="callback", callback=getrelationships, dest="output")
parser.add_option("-q", "--question", help="Classify questions", action="callback", callback=getquestiontype, dest="output")

(options, args) = parser.parse_args()
print(options.output)
