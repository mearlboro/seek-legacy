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

import os, glob, sys, pickle, re
import string
from operator  import *
from itertools import *
from functools import *
from optparse import OptionParser

import numpy, nltk, gensim
# from nltk.corpus import stopwords
from gensim import corpora, models, similarities

import logging
logger = logging.getLogger('handler')

# ------------------------------------------------------------------------------------
# spacy.io
# ------------------------------------------------------------------------------------
print("Loading dependencies, please wait")
from spacy.en import English
from spacy import attrs
from spacy.tokens import Doc
nlp = English()
print("Dependencies have been loaded")
# ------------------------------------------------------------------------------------

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
    voc = sorted(set([tok.text for tok in toks]))
    # voc = sorted(set(voc + sorted(set([w.lower() for w in toks]))))
    # print(voc)
    return voc

# Get the word frequencies of a set of tokens.
# The results of this function can be simply added for multiple texts.
def word_freq(toks):
    word_freqs = {}
    toks = filter_punct(toks)
    toks = nlp(' '.join(toks))

    filtered_toks = nlp(' '.join(filter_stop_words(toks)))

    freqs = filtered_toks.count_by(attrs.ORTH)
    for freq in freqs.items():
        word_freqs[nlp.vocab.strings[freq[0]]] = freq[1]
    return word_freqs

# Filter punctuation.
def filter_punct(toks):
    words = [tok.text for tok in toks if not tok.is_punct]
    return words

# Filter out stop words and irrelevant parts of speech from a set of tokens.
def filter_stop_words(toks):
    corpus_filter = [tok.text for tok in toks if not tok.is_stop]
    return corpus_filter

def normalize(toks):
    for tok in toks:
        print(tok.lemma_)
        print(tok.pos_)
    return [tok.lemma_ for tok in toks]

# Get the weight of each sentence in a text based on frequency.
def sentence_freq(tokens, sents):
    # get and filter words
    # words = nltk.word_tokenize(text)
    # tokens = nlp(text)
    tokens = nlp(' '.join(filter_punct(tokens)))
    filtered_words = nlp(' '.join(filter_stop_words(tokens)))

    # get vocab and freqs
    voc = vocab(filtered_words)
    freqs = word_freq(filtered_words)
    
    # when summing frequencies per sentence thus use wordfreqs
    sentfreqs = []
    for sent in sents:
        sentfreqs += [(sent.text, numpy.mean([freqs[word.text] if word.text in voc else 0 for word in sent]))]
    return sentfreqs



###########################################################################################

# -- COMMAND mostfreq ---------------------------------------------------------------------
def getmostfreq(option, opt_str, value, parser):
    args = parser.rargs
    src = args[0]
    pretty = args[1]
    if not pretty:
        print("The most frequent word is ...")

    docs = getdocs(src, pretty)
    count = 0
    words = []
    freqs = []
    sortedfreqs = []
    for doc in docs:
        tokens = nlp(doc)
        words.append(tokens)
        freqs.append(word_freq(words[count]))
        sortedfreqs.append(sorted(freqs[count].items(), key=lambda x:x[1], reverse=True))
        count += 1
    if not pretty:
        setattr(parser.values, option.dest, sortedfreqs)
    else:
        setattr(parser.values, option.dest, prettifymostfreq(sortedfreqs))



# -- COMMAND summary ---------------------------------------------------------------------
'''
When summing frequencies per sentence add bias from topics in that phrase
    model: 0 for LDA, 1 for LSI
    text : the content of a document in a string
    sents: the sentences in a document - a list of lists of word and punctuation tokens
    freqs: the output of sentence_frequency(text, sents)
'''
def augment_topics(model, tokens, sents, freqs):
    num = 10

    if model == "LDA":
        topics = lda2dict(lda([tokens.text], num))[1]
    elif model == "LSI":
        topics = lsi2dict(lsi([tokens.text], num))
    else:
        print("Unrecognized model: " + model)
        sys.exit(0)

    sentfreqs = []
    for sent, freq in freqs:
        sentfreqs +=  [(''.join(sent),
                        freq + sum([word if word.lower() in topics.keys() and topics[word.lower()] else 0 for word in sent])
                      )]
    return sentfreqs

'''
When summing frequencies per sentence add bias from named entities in that phrase
    text : the content of a document in a string
    sents: the sentences in a document - a list of lists of word and punctuation tokens
    freqs: the output of sentence_frequency(text, sents)
'''
def augment_ne(tokens, sents, freqs):
    bias = 2

    nes = ne([tokens.text]) # get single-word NEs by means of text analysis
    nes = [ne[0] for ne in nes] # the words as list

    sentfreqs = []
    for sent,freq in freqs:
        sentfreqs +=  [(''.join(sent), 
                        freq * sum([word if word.lower() in nes and bias else 1 for word in sent])
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
        print("Incorrect arguments:\nUsage: linguist.py --summary src model no_sentences pretty")
        sys.exit(0)
    src = args[0]
    model = args[1]
    num = int(args[2])
    pretty = int(args[3])

    text = "Constructing summary for documents at {} ..."
    print(text.format(src))

    docs = getdocs(src, pretty)

    summaries = []
    for doc in nlp.pipe(docs, n_threads = 4):
        sents = doc.sents
        freqs = sentence_freq(doc, sents)
        if model in ["LDA", "LSI"]:
            freqs = augment_topics(model, doc, sents, freqs)
        elif model == "NE":
            freqs = augment_ne(doc, sents, freqs)
        sortedfreqs = sorted(freqs, key=lambda x: x[1], reverse=True)
        min_freq = sortedfreqs[num][1]
        summary = [f[0] for f in freqs if f[1] >= min_freq]

        summaries += [summary]

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

def ne(docs):
    entities = []

    for doc in nlp.pipe(docs, n_threads = 4):
        nes = [(ent, ent.label_) for ent in doc.ents]
        entities += [nes]

    return entities

def getentities(option, opt_str, value, parser):
    args = parser.rargs
    if len(args) < 3:
        print("Incorrect arguments:\nUsage: linguist.py --entities src ne_type pretty")
        sys.exit(0)
    src = args[0]
    ntype = args[1]
    pretty = int(args[2])
    if ntype not in ["PERSON", "GPE", "DATE", "ORG", "ALL"]:
        print("Incorrect arguments: ne_type can be PERSON, GPE, DATE, ORG or ALL")
    type_name = ntype.lower()
    if not pretty:
        out_text = "Retrieving {} named entities for documents at ..."
        print(out_text.format(type_name, src))

    docs = getdocs(src, pretty)
    nes = ne(docs)
    if ntype == "ALL":
        selected_nes = nes
    else:
        selected_nes = [[n for n in nesdoc if n[1] == ntype] for nesdoc in nes]

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
        print("Incorrect arguments:\nUsage: linguist.py --topics src model no_topics pretty")
        sys.exit(0)
    src = args[0]
    model = args[1]

    num = int(args[2])
    pretty = int(args[3])
    if not pretty:
        print("Retrieving topics by the " + model + " model for documents at " + src + " ...")

    docs = getdocs(src, pretty)
    topics = []

    if model == "LSI":
        topics.append(lsi2dict(lsi(docs, num)))
    elif model == "LDA":
        topics.append(lda2dict(lda(docs, num))[0])
    else:
        print("Unrecognized model: " + model)
        sys.exit(0)

    if not pretty:
        setattr(parser.values, option.dest, topics)
    else:
        setattr(parser.values, option.dest, prettifytopics(topics))

def lsi2dict(topics):
    topics = topics[0][1].split('+')
    pairs  = [topic.split('*') for topic in topics]
    pairs  = [(''.join(list(filter(lambda c:c not in "\" ", nlp(pair[1])[0].lemma_))), float(pair[0])) for pair in pairs]
    return dict(pairs)

def lda2dict(topics):
    dicts = []
    for i in range(len(topics)):
        topic = topics[i][1].split('+')
        pairs = [t.split('*') for t in topic]
        pairs = [(''.join(list(filter(lambda c:c not in " ", nlp(pair[1])[0].lemma_))), float(pair[0])) for pair in pairs]
        dicts += [dict(pairs)]

    return dicts


'''
Get lsi and lda topics
'''
def lsi(docs, num):
    # tokenize each doc, filter punctuation and stop words
    # docs = list(map(filter_punct, map(nltk.word_tokenize, docs)))
    # filtered = [f[1] for f in list(map(filter_stop_words, docs))]
    tokens = [nlp(doc) for doc in docs]
    tokens = [nlp(' '.join(filter_punct(toks))) for toks in tokens]
    filtered = [filter_stop_words(toks) for toks in tokens]

    # create Gensim dictionary and corpus
    dictionary = corpora.Dictionary(filtered) # choose the text with lowercase words
    corp = [dictionary.doc2bow(reduce(add, filtered))]

    lsi_topics = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=dictionary, num_topics=num)

    # returns the topics as a dictionary of words and scores
    return lsi_topics.print_topics(num)


def lda(docs, num):
    # tokenize each doc, filter punctuation and stop words
    # docs = list(map(filter_punct, map(nltk.word_tokenize, docs)))
    # filtered = [f[1] for f in list(map(filter_stop_words, docs))]
    tokens = [nlp(doc) for doc in docs]
    tokens = [nlp(' '.join(filter_punct(toks))) for toks in tokens]
    filtered = [filter_stop_words(toks) for toks in tokens]
    # doc = ' '.join(filtered)
    # print(filtered)
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

    # nes_merged = dict([(' '.join(n[0]), n[1]) for n in nes])
    nes_merged = nes
    pers_org = dict(filter(lambda t: t[1] == 'PERSON', nes_merged))
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

def _span_to_tuple(span):
    start = span[0].idx
    end = span[-1].idx + len(span[-1])
    tag = span.root.tag_
    text = span.text
    label = span.label_
    return (start, end, tag, text, label)

def merge_spans(spans, doc):
    # This is a bit awkward atm. What we're doing here is merging the entities,
    # so that each only takes up a single token. But an entity is a Span, and
    # each Span is a view into the doc. When we merge a span, we invalidate
    # the other spans. This will get fixed --- but for now the solution
    # is to gather the information first, before merging.
    tuples = [_span_to_tuple(span) for span in spans]
    print(tuples)
    for span_tuple in tuples:
        doc.merge(*span_tuple)

def merge_ents(doc):
    for ent in doc:
        if ent in doc.ents:
            print(ent)

def getrelationships(option, opt_str, value, parser):
    args = parser.rargs
    if len(args) < 2:
        print("Incorrect arguments:\nUsage: linguist.py --relationships src pretty")
        sys.exit(0)
    src = args[0]
    pretty = int(args[1])
    print("Extracting information from documents at " + src + " ...")

    docs = getdocs(src, pretty)

    dbs = []
    nes = ne(docs)
    for doc in nlp.pipe(docs, n_threads = 4):
        # ents = [(ent, ent.label_) for ent in doc.ents]
        # print(ents)
        # merge_spans(doc.ents, doc)
        # merge_spans(doc.noun_chunks, doc)
        # ents = [(ent, ent.label_) for ent in doc.ents]
        # print(ents)
        db = {}
        # Construct a dictionary of the form: Value of NE, relation, [(attribute, NE tag)]
        # sents  = st.text2sents(doc)
        sents = doc.sents
        # sents  = [list(filter(lambda x: x not in string.punctuation, sent)) for sent in sents]
        ldas   = [l for l in lda2dict(lda([doc.text], 2))[0]]
        chunks = list(doc.noun_chunks)
        # tokens = nlp(' '.join(filter_punct(doc)))
        # filtered_words = nlp(' '.join(filter_stop_words(tokens)))
        # print(doc.ents)
        for ent in doc:
            merge_spans(doc.ents, doc)
            merge_spans(doc.noun_chunks, doc)
            
            # Joining Barack with Obama, United with States
            # if ent.dep_ in ('compound'):
                # print(ent.text + ' ' + ent.head.text)


            if ent.pos_ in ('VERB') and ent.dep_ in ('ROOT', 'conj'):
                relation = [w for w in ent.head.lefts if w.pos_ in ('VERB', 'aux')] + [ent]
                print(relation)
            if ent.dep_ in ('attr', 'dobj', 'pobj', 'npadvmod'):
                subject = [w for w in ent.lefts if w.dep_.startswith('nsubj')]
                if subject:
                    subject = subject[0]
                    if subject in db.keys():
                        db[subject].append(ent)
                    else:
                        db[subject] = [ent]
                elif ent.dep_ == 'pobj' and ent.head.dep_ == 'prep' or ent.dep_ == "ROOT" and ent.head.dep_ == 'aux':
                    if ent.head.head in db.keys():
                        db[ent.head.head].append(ent)
                    else:
                        db[ent.head.head] = [ent]
            for k in db.keys():
                for l in db[k]:
                    if l in db.keys():
                        db[k] += db[l]
                        # db.pop(l, 'None')
            for k in db.keys():
                for l in db[k]:
                    if l in db.keys():
                        db[l] = []
            rel = {}
            for k in db.keys():
                if db[k] != []:
                    rel[k] = db[k]
        dbs += [rel]
        # for sent in sents:
            # print(sent)
            # for tok in sent:
                # print(tok.text + " <==== " + tok.head.text)
                # print(tok.tag_ + " <==== " + tok.head.tag_)
                # print(list(tok.children))
        # for fw in sents:
            # print(fw.text + " <==== " + fw.head.text)
            # print(list(fw.lefts))
            # print(list(fw.rights))
            # print(list(fw.subtree))
            # print(fw.text + " <----- " + fw.head.text)
            # print(list(fw.children))
        # nes = doc.ents
        # ats, rels = relations(sents, chunks, nes, ldas)
        # nes = dict([(' '.join(n[0]), n[1]) for n in nes])
        # for ent in ats.keys():
        #     prev_rel = None
        #     index = 1
        #     if ent in rels.keys():
        #         for relation in rels[ent]:
        #             attributes = []
        #             index += 1
        #             if relation != "":
        #                 prev_rel = relation
        #                 for atrb in ats[ent][0:index]:
        #                     if atrb in nes.keys():
        #                         attributes.append((atrb, nes[atrb]))
        #                     elif ent in nes.keys():
        #                         attributes.append((atrb, nes[ent]))
        #                 if ent in nes.keys():
        #                     dbs += [((ent, nes[ent]), prev_rel, attributes)]
        #                 else:
        #                     words = ent.split()
        #                     for word in words:
        #                         if word in nes.keys():
        #                             dbs += [((ent, nes[word]), prev_rel, attributes)]
        #                 attributes = []
        #                 del ats[ent][0:index]
        #                 index = 1
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
parser.add_option("-e", "--entities", help="Extract named entities. Required src topic_model entity_type pretty", action="callback", callback=getentities, dest="output")
parser.add_option("-s", "--summary", help="Offer summary of text. Required src, topic_model, no_sentences, pretty", action="callback", callback=getsummary, dest="output")
parser.add_option("-t", "--topics", help="Extract topics from text. Required src, topic_model, no_topics, pretty", action="callback", callback=gettopics, dest="output")
parser.add_option("-r", "--relationships", help="Extract relationships from text. Required src pretty", action="callback", callback=getrelationships, dest="output")
parser.add_option("-q", "--question", help="Classify questions. Required question in string format", action="callback", callback=getquestiontype, dest="output")
parser.add_option("-f", "--freq", help="Return most frequent word. Required src pretty", action="callback", callback=getmostfreq, dest="output")

(options, args) = parser.parse_args()
print(options.output)
