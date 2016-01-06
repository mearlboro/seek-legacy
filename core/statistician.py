# -*- coding: utf-8 -*-
import sys
import os
import string
import logging
import pprint
import json
import regex as re
from datetime import datetime
from functools import reduce
from operator import add

import nltk
import nltk.chunk
import nltk.data
import nltk.tag
from nltk.collocations import *
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramAssocMeasures
from nltk.tokenize import MWETokenizer
import gensim
from gensim.corpora import WikiCorpus, wikicorpus

# -- SENTENCE TOKENIZER ---------------------------------------------------
'''
"NLP with Python" book, chapter 6.2

The sentence tokenizer uses a custom regex tokenizer to split a given text into words,
punctuation, and spaces (WordPunctSpaceTokenizer).
Having the tokens as described above, it uses the punctuation_features feature extractor
for identifying features that would help identify whether at that specific token, given
that it is a punctuation mark from the set ".!?", a sentence ends and another starts.
The classifier based on these features is trained on the NLTK Treebank corpus of already
tokenized sentences.

The API consists of:
text2sents(text):
    gets a text in string format, splits it into tokens, then classifies them into sentences
    input:   a text in unicode/string format
    returns: a list of lists of tokens in unicode/string format, representing the sentences

'''

# Tokenize text into words, punctuation, and spaces
class WordPunctSpaceTokenizer(nltk.tokenize.RegexpTokenizer):
    def __init__(self):
        nltk.tokenize.RegexpTokenizer.__init__(self, r'\w+|\s+|\.+|[\-\\=]+|[^\w\s]')

# Tokenize text into sentences
class SentenceTokenizer():
    # extract punctuation features from token list for punctuation sign (token of index i)
    def __punctuation_features(self, toks, i):
        return {
            'punct': toks[i],
            'is-next-capitalized': (i < len(toks) - 1) and toks[i+1][0].isupper(),
            'lower-or-punct-prev': toks[i-1].lower() or toks[i-1] in string.punctuation,
            'is-prev-one-char': len(toks[i-1]) == 1
        }

    # Builds the classifier
    def __init__(self):
        print(str(datetime.now()) + ": Training sentence tokenizer decision tree classifier on Treebank corpus...")

        # use the simple tokenizer to get words, punctuation, whitespace
        self.tokenizer = WordPunctSpaceTokenizer()

        # join the sentence corpus into a text
        training_sents = nltk.corpus.treebank_raw.sents()
        toks = []
        bounds = set()
        offset = 0
        for sent in training_sents:
            toks.extend(sent)  # union of toks in all sentences
            offset = offset + len(sent)
            bounds.add(offset-1) # known boundaries of sentences

        # Create training features by calling punctuation_features on sentence delimiters {'.', '?', '!'}
        featuresets = [(self.__punctuation_features(toks,i), (i in bounds))
                       for i in range(1, len(toks)-1)
                       if toks[i] in '.?!']

        # Decision Tree classifier for training with the Treebank corpus
        size = int(len(featuresets)*0.2)
        train_set, test_set = featuresets[size:], featuresets[:size]
        self.classifier = nltk.DecisionTreeClassifier.train(train_set)

        print(str(datetime.now()) + ": Classifier trained with accuracy " + str(nltk.classify.accuracy(self.classifier, test_set)))


    # Use the classifier to segment word toks into sentences
    def __classify_sentences(self, words):
        start = 0
        sents = []
        for i in range(len(words)):
            if words[i] in '.?!' and self.classifier.classify(self.__punctuation_features(words,i)) == True:
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            sents.append(words[start:])
        return sents


    # Segment text into sentences and words
    def text2sents(self, text):

        # turn whitespace characters into spaces: split() runs on whitespace then merge words back with spaces
        text = ' '.join(text.split())

        # tokenise with the Regexp tokenizer thus keeping the punctuation, words, and spaces
        toks = self.tokenizer.tokenize(text)
        # filter out irrelevant punctuation
        toks = list(filter(lambda tok: tok not in '"()[]{}', toks))

        # Create list of sentences using the classifier, then iterate through words in a sentence to collapse abbreviations into single words
        sentences = []
        for sent in self.__classify_sentences(toks):
            sentence = []
            i = 0
            tok = ""
            for word in sent:
                i = sent.index(word)
                if (word[0] in string.punctuation and not word[0] in '.?!'):
                    # punctuation that should be kept
                    if (len(tok) > 0):
                        sentence.append(tok)
                        tok=""
                    sentence.append(word)
                elif (word.isspace()):
                    # space character - finish a word token
                    if (len(tok) > 0):
                        sentence.append(tok)
                        tok = ""
                elif (i == len(sent)-2):
                    # penultimate end of the sentence - break off the punctuation
                    sentence.append(tok+word)
                    tok = ""
                else:
                    # accumulate a token in tok
                    tok = tok + word

            # Add tok to the current sentence
            if len(tok) > 0:
                sentence.append(tok)
            sentences.append(sentence)

        return sentences




# -- MULTI WORD EXPRESSIONS TOKENIZER ------------------------------------

'''
using dictionaries found at mwe.stanford.edu/resources

The API consists of:
words2mwes(words):
    gets a list of words in string format and classifies them where necessary into expressions
    input:   a list of words in unicode/string format
    returns: a list of tokens in unicode/string format, either single words or expressions
'''

class SharoffMWETokenizer():
    # Helper function to generate the n-grams using chi-square test from the treebank
    # ngram is a function: nltk.bigram or nltk.trigram
    # AssocMeasures is a class: BigramAssocMeasures or TrigramAssocMeasures
    # CollocationFinder is a class: BigramCollocationFinder or TrigramCollocationFinder
    def __get_ngrams(self, training_sents, ngram, AssocMeasures, CollocationFinder):
        # to create sets of examples we use the bigrams we find in the training sentences
        ngrams = list(map(ngram, training_sents))
        ngrams = list(map(list, ngrams)) # unwrap ngrams generator objects

        ngram_measures = AssocMeasures() # will compute chi-square test for all ngrams
        finder = CollocationFinder.from_words(
            nltk.corpus.treebank_raw.words(),
            window_size = 20)
        # a list of collocation generator objects to be identified based on chi-square test
        found = list(map(lambda x: finder.above_score(ngram_measures.raw_freq, 1.0 / x), map(len, tuple(ngrams))))
        found =  reduce(add, map(list, found)) # reduce it to the final list of collocation (warning: slow)

        return (ngrams, found)


    # sharoff dictionary consists of expressions and their statistical collocation measures
    # the feature extractor below grabs these features from the dictionary
    def __Sharoff_features(self, expr):
        return {
            'expr': expr,
            'T-score':  self.SharoffDict[expr]['T']  or 0,
            'MI-score': self.SharoffDict[expr]['MI'] or 0,
            'LL-score': self.SharoffDict[expr]['LL'] or 0,
            'number-of-words': len(expr)
        }


    # To train the classifier we use the Sharoff features on bigrams found in treebank
    # and also in the Sharoff Dictionary.
    # To create training examples, we use the features of bigrams above and for the
    # targets whether they are in the set of bigram collocations in treebank corpus
    # generated with the Chi-square test provided by nltk's AssocMeasures()
    def __init__(self):
        print(str(datetime.now()) + ": Training decision tree classifier for the multi word expression tokenizer based on Sharoff's dictionary on the Treebank corpus...")

        # get the Sharoff dictionary
        f = open("../dictionaries/sharoff.json", 'r+')
        self.SharoffDict = json.load(f)
        f.close()

        # get the traininf corpus: join the treebank sentence corpus into a text
        # and filter out START tag and punctuation
        training_sents = [list(filter(lambda w: w not in ['START'] and w not in string.punctuation, sent))
                          for sent in nltk.corpus.treebank_raw.sents()]
        # filter out empty or 1-word sentences
        training_sents = list(filter(lambda s: len(s) > 1, training_sents))
        toks = reduce(add, training_sents) # merge all sentences into one text of tokenized words

        # get all bigrams and trigrams in training_sents and all statistical bigram and trigram collocations found in treebank
        (bigrams, foundbigrams)   = self.__get_ngrams(training_sents, nltk.bigrams,  BigramAssocMeasures,  BigramCollocationFinder)
        (trigrams, foundtrigrams) = self.__get_ngrams(training_sents, nltk.trigrams, TrigramAssocMeasures, TrigramCollocationFinder)

        # Create training examples with sharoff features, for each ngram in the dictionary
        examples = [(self.__Sharoff_features(expr), expr in foundbigrams) #or expr in foundtrigrams)
                       for expr in bigrams #or expr in trigrams
                       if expr in self.SharoffDict.keys()]

        #  classifier for training with the Treebank corpus
        size = int(len(examples)*0.2)
        train_set, test_set = examples[size:], examples[:size]
        self.classifier = nltk.DecisionTreeClassifier.train(train_set)

        print(str(datetime.now()) + ": Classifier trained with accuracy " + str(nltk.classify.accuracy(self.classifier, test_set)))



    # Use the classifier defined above to segment word toks into MWEs
    def words2mwes(self, words):
        start = 0
        toks = []
        for ngram in list(nltk.bigrams(words)) + list(nltk.trigrams(words)):
            if self.classifier.classify(self.__Sharoff_features(ngram)) == True:
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            toks.append(words[start:])
        return toks




# -- MULTI WORD EXPRESSIONS TOKENIZER -----------------------------------------

'''
using dictionaries found at mwe.stanford.edu/resources

The API consists of:
words2mwes(words):
    gets a list of words in string format and classifies them where necessary into expressions
    input:   a list of words in unicode/string format
    returns: a list of tokens in unicode/string format, either single words or expressions
'''

class McCarthyMWETokenizer(MWETokenizer):
    # this chunker is static and uses a list of 116 frequent phrases found by McCarthy
    # with the nltk standard MWE chunker

    def __init__(self):
        # get the McCarthy dictionary
        f = open("../dictionaries/mccarthy.json", 'r+')
        self.McCarthyDict = json.load(f)
        f.close()

        self.__tokenizer = MWETokenizer()
        for key in self.McCarthyDict.keys():
            self.__tokenizer.add_mwe(nltk.word_tokenize(key))


    # use the tokenizer to segment words into MWEs. Default behaviour of MWE tokenizer
    # adds an underscore between MWE words thus this function removes it
    def words2mwes(self, words):
        toks = [tok.replace('_', ' ') for tok in self.__tokenizer.tokenize(words)]
        return toks




# -- CHUNK PARSER -----------------------------------------------------------
'''
splits sentences in noun and verb phrases and other sentence structures

The API consists of:
text2chunks(text):
    gets a text in string/unicode format and classifies to sentences split in chunks
    input:   a text in unicode/string format
    returns: a list of chunk trees representing the phrasal chunks of each sentence in the text
'''

class ChunkParser():
    def __init__(self):
        self.chunker = nltk.data.load("chunkers/treebank_chunk_ub.pickle")
        # self.tagger = nltk.data.load("taggers/brown_aubt.pickle")

    def text2chunks(self, text):
        sent_tok = SentenceTokenizer()
        tokenized_sentences = sent_tok.text2sents(text)
        # Chose nltk.pos_tag for simplicty. For more complex answers, try brown
        # TODO: train chunker on brown if that's the case
        # tagged_sentences = list(map(self.tagger.tag, tokenized_sentences))
        tagged_sentences = list(map(nltk.pos_tag, tokenized_sentences))
        chunked_sentences = list(map(self.chunker.parse, tagged_sentences))
        return chunked_sentences




# -- TOPIC MODELLING ----------------------------------------------------------

class TopicModelling():
    def __init__(self):
        wiki_src = '../raw/wiki/enwiki-articles.xml.bz2'

        # load the corpus of documents in the wikipedia archive and save parsed files to disk
        self.wiki_corpus = WikiCorpus(wiki_src)
        self.wiki_dictionary = self.wiki_corpus.dictionary
        self.wiki_dictionary.save("../raw/wiki/parsed/wiki_dict.dict")
        MmCorpus.serialize("../raw/wiki/parsed/wiki_corpus.mm")
      
        
    # extract topics with lda
    # lda_text: tokenized text that has already been processed for stopwords, collocations, MWEs, normalization etc
    # num:      number of topics to extract
    def text2ldatopics(self, lda_text, num):
        corp = [self.wiki_dictionary.doc2bow(lda_text)]

        lda_topics = gensim.models.ldamodel.LdaModel(corpus=corp, id2word=self.wiki_dictionary, num_topics=num)

        print(lda_topics.print_topics(num))
        return lda_topics

    # extract topics with lsi
    # lda_text: tokenized text that has already been processed for stopwords, collocations, MWEs, normalization etc
    # num:      number of topics to extract
    def text2lsitopics(self, lsi_text, num):
        corp = [self.wiki_dictionary.doc2bow(lsi_text)]

        lsi_topics = gensim.models.lsimodel.LsiModel(corpus=corp, id2word=self.wiki_dictionary, num_topics=num)

        # returns the topics as a dictionary of words and scores
        topics = lsi_topics.print_topics(num)[0][1].split('+')
        pairs = [topic.split('*') for topic in topics]
        pairs = [(''.join(list(filter(lambda c:c not in "\" ", pair[1]))), float(pair[0])) for pair in pairs]
        return dict(pairs)



# -- NAME ENTITY DETECTOR ---------------------------------------------------
'''
extracts named entities from an input text

The API consists of:
text2ne(text):
    gets a text in string/unicode format and extracts named entities by mapping them
    against the chunked sentences of the text and grouping them based on NP chunks
    input:   a text in unicode/string format
    returns: a list of tuples of the format (NE, TYPE)
'''

class NameEntityDetector():
    def __init__(self):
        self.chunker = ChunkParser()
        self.stanford_tagger = nltk.tag.StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

    def text2ne(self, input_text):
        chunked_sents = self.chunker.text2chunks(input_text)
        named_entities = dict(self.stanford_tagger.tag(re.split("\,?\.?\s+", input_text)))
        # Create a list to store the final mapping of NEs
        answered = []
        for chunked_sent in chunked_sents:
            # print(chunked_sent)
            filtered_chunked_subtrees = chunked_sent.subtrees(filter= lambda t: t.label() == 'NP')
            for subtree in filtered_chunked_subtrees:
                category = None
                ent_key = ""
                for leaf in subtree.leaves()[:-1]:
                    if (leaf[0] in named_entities):
                        ent_key += leaf[0] + " "
                        category = named_entities[leaf[0]]
                if (subtree.leaves()[-1][0] in named_entities):
                    ent_key += subtree.leaves()[-1][0]
                    if (category is None):
                        category = named_entities[subtree.leaves()[-1][0]]
                if(category is None):
                    category = 'O'
                answered += [(ent_key, category)]
        return set(filter(lambda x: x[1] != 'O', answered))

    def text2unine(self, input_text):
        named_entities = dict(self.stanford_tagger.tag(re.split("\,?\.?\s+", input_text)))
        return named_entities 

ned = NameEntityDetector()

f = open(sys.argv[1])
input_text = f.read()
print(ned.text2ne(input_text))
