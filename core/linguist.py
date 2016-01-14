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
            documents += [open(f, 'r+').read()]
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



#######################################################################################

# -- COMMAND summary ---------------------------------------------------------------------
'''
When summing frequencies per sentence add bias from topics in that phrase
    model: 0 for LDA, 1 for LSI
    text : the content of a document in a string
    sents: the sentences in a document - a list of lists of word and punctuation tokens
    freqs: the output of sentence_frequency(text, sents)
'''
def augment_topics(model, text, sents, freqs):
    num = 10 # TODO: choose a number that has a relevance!!!

    if model == 0:
        topics = lda2dict(lda([text], num))[1] # TODO: better idea?
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
    bias = 2 # TODO: meaningful number

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
def getsummary(src, args):
    if len(args) < 2:
        print("Incorrect arguments: expected \n linguist.py summary <src> <model> <num>")
        sys.exit(0)

    print("Constructing summary for documents at " + src + " ...")

    model = args[0]
    num = args[1]
    docs = getdocs(src)

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

    return summaries




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

    entities = []
    for doc in docs:
        sents  = st.text2sents(doc)
        if model == 0:
            chunks = ch.sents2chunks(sents)
            nes    = ner.chunks2ne(doc, chunks)
        elif model == 1:
            nes    = ner.text2ne(doc)

    del st
    del ch
    del ner

    return nes

def getentities(src, args):
    if len(args) < 2:
        print("Incorrect arguments: expected \n linguist.py summary <src> <model> <num>")
        sys.exit(0)

    model = args[0]
    model_name = 'chunk-based' if 0 else 'text-based'
    print("Retrieving named entities by " + model_name + " detection for documents at " + src + " ...")

    docs  = getdocs(src)

    return ne(docs, model)



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

    model = args[0]
    model_name = 'LDA' if 0 else 'LSI'
    print("Retrieving topics by the " + model_name + " model for documents at " + src + " ...")

    docs = getdocs(src)

    if model == 1:
        print("LSI model topics:")
        return lsi(docs, args[1])
    else:
        print("LDA model topics:")
        return lda(docs, args[1])


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


def lsi(docs, num):
    # TODO: chunk mwes and collocations if necessary.
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
    # TODO: chunk mwes and collocations if necessary.
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


def attribs(sents, chunks, nes, ldas):

    adjs = [ 'JJ', 'JJR', 'JJS' ]
    vbs  = [ 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' ]
    nns  = [ 'NN', 'NNS', 'NNP', 'NNPS' ]
    ins  = [ 'of', 'that', 'which', 'like', 'in', 'at', 'as' ]

    nes_merged = dict([(' '.join(n[0]), n[1]) for n in nes])
    pers_org = dict(filter(lambda t: t[1] == 'PERSON' or t[1] == 'ORGANIZATION', nes_merged.items()))
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # dictionary of dictionaries for each named entity
    retrieved = {}
    # retrieved = []
    for chunked_sent in chunks:
        # filter through all chunks for NPs, get each subtree that may contain attributes next to nouns
        # then create dictionary entry for the named entity/noun if it does not exist
        # and add the adjectives in the bag
        filtered_chunked_subtrees = list(chunked_sent.subtrees(filter= lambda t: t.label() == 'NP')) # or any(map(lambda l: l[1] in nns, t.leaves())))
        # merge noun that comes after noun phrase into a noun phrase
        relation = []
        prev_ne = None
        for subtree in filtered_chunked_subtrees:
            # print(subtree)
            ent_key = ' '.join(list(map(lambda t: t[0], subtree.leaves())))
            if (any(word in ent_key for word in pers_org.keys())):
                prev_ne = ent_key
            else:
                relation.append(ent_key)
            # ind = chunked_sent.index(subtree)
            # if (ent_key not in retrieved):
            # if (ind < len(chunked_sent) - 1):
                # if (chunked_sent[ind + 1][1] in vbs):
            if (len(relation) >= 2 and prev_ne != None):
                # retrieved.append(relation)
                # relation = []
                retrieved[prev_ne] = relation
                # if (pers_org[prev_ne] == "PERSON"):
                #     for name in prev_ne.split(" "):
                #         retrieved[name] = relation
                relation = []
                # for porg in pers_org:
                    # if (porg in " ".join(relation)):
                        # retrieved[porg]= dict((porg, relation))
                    # relation = []
            #         ind = chunked_sent.index(subtree) + 1
            #         relation = []
            #         while (ind < len(chunked_sent)):
            #             if (type(chunked_sent[ind]) is tuple):
            #                 relation.append([chunked_sent[ind][0]])
            #             else:
            #                 words = list(map(lambda t: t[0], chunked_sent[ind].leaves()))
            #                 not_ent = ' '.join(words)
            #                 if (not_ent not in nes_merged.keys()):
            #                     relation.append(words)
            #             ind += 1
            #         retrieved.append(ent_key)
            #         retrieved.append(' '.join(sum(relation, [])))
            #         if (type(chunked_sent[ind - 1] is tuple))
            #         retrieved.append(' '.join(list(map(lambda t: t[0], chunked_sent[ind - 1].leaves()))))
            # else:
            #     continue
                # print(chunked_sent[ind + 1])
        # for t in filtered:
        #     ind = chunked_sent.index(t)
        #     # print(ind)
        #     if ind < len(chunked_sent) - 1:
        #         nextt = chunked_sent[ind + 1]
        #     if t.label() == 'NP' and nextt in nns:
        #         for ne in nes_merged:
        #             if (t)
                # t[ind] = nltk.tree.Tree('NP', [ind, nextt])
                    # print(t[ind])
                # del t[ind + 1]
    #     grammar = '''
    #         ADJ:  {<IN><JJ.*>*<NP>}
    #         ATTR: {<NP><IN><NP>}
    #         '''
    #     regex_chunker = nltk.RegexpParser(grammar) # will split words into groups as in grammar
    #
    #     for t in filtered_chunked_subtrees:
    #         if t.label() == 'NP':
    #             words = [l[0] for l in t.leaves()]
    #             pos = nltk.pos_tag(words)
    #             regex_chunks = regex_chunker.parse(pos)
    #             print(regex_chunks)
    #             # the first case in the grammar, e.g. 'beautiful Hawaii'
    #             for subt in regex_chunks.subtrees():
    #                 subject = ''
    #                 if subt.label() == 'NP' or subt.label in nns:
    #                     # if ne or topic, then it becomes node
    #                     print(subt)
    #                     possible_subject = ' '.join([l[0] for l in subt.leaves()])
    #                     print(possible_subject)
    #                     if possible_subject in [n[0] for n in nes_merged] or possible_subject in ldas:
    #                         subject = possible_subject
    #                         subject_type = nes_merged.get(subject)
    #
    #                 attributes_bag = []
    #                 if subt.label() == 'JJ':
    #                     # grab all adjectives
    #                     attributes_bag += [' '.join([l[0] for l in subt.leaves()])]
    #                 print(attributes_bag)
    #
    #             # now add findings to dictionary
    #             if subject != '':
    #                 if subject in retrieved.keys():
    #                     if subject_type in retrieved[subject].keys():
    #                         retrieved[subject][subject_type] += attributes_bag
    #                     else:
    #                         retrieved[subject] = dict((subject_type, attributes_bag))
    #                 else:
    #                     retrieved[subject] = dict((subject_type, attributes_bag))
    #
    return retrieved


'''
Barack Obama is the prezident of the United States.

NP Barack Obama
VP is
NP the prezident of the United States

VP = is, was, etc.

is(barack, prezinf)

person(barack, prezindent of the united states, born in hawaii
'''
def relations(sents, chunks, nes, ldas):
    return {}

def getrelationships(src, args):
    print("Extracting information from documents at " + src + " ...")

    docs = getdocs(src)

    st  = getSentenceTokenizer()
    ch  = getChunkParser()
    ner = getNameEntityDetector()

    dbs = []

    for doc in docs:
        db = {}

        sents  = st.text2sents(doc)
        sents  = [list(filter(lambda x: x not in string.punctuation, sent)) for sent in sents]
        ldas   = [l for l in lda2dict(lda([doc], 2))[0]]
        nes    = [ne[0] for ne in ner.text2ne(doc)]
        sents  = list(filter(lambda sent: any([t in sent for t in ldas]) or any([ne in sent for ne in nes]), sents))
        chunks = ch.sents2chunks(sents)
        nes    = ner.chunks2ne(doc, chunks)

        # TODO: this is actually relations, to be updated

        ats = attribs(  sents, chunks, nes, ldas)
        print(ats)
        # attribs(  sents, chunks, nes, ldas)
        # rls = relations(sents, chunks, nes, ldas)

        db = ats
    dbs += [db]

    return dbs



# -- COMMAND questions ------------------------------------------------------------------
'''
Classifies questions based on the question classifier
    <text> is the question in string format
    takes no args
'''
def getquestiontype(text, args):

    qc  = getQuestionClassifier()
    ner = getNameEntityDetector()

    nes = ner.text2ne(text)
    c = qc.classify(text, nes)

    del qc
    del ner

    return c

##########################################################################################

commands = {
    'summary': getsummary,
    'entities': getentities,
    'topics': gettopics,
    'relationships': getrelationships,
    'question': getquestiontype,
    # 'similar': todo
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
        print("<command> can be \n summary \n entities \n topics \n relationships \n question ")
        sys.exit(0)
