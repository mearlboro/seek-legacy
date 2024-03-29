# -*- coding: utf-8 -*-
import sys, os, string, logging, pprint, json, subprocess
import regex as re
from datetime import datetime
from functools import reduce
from operator import add

import nltk
import  nltk.chunk, nltk.data, nltk.tag
from nltk.collocations import *
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures
from nltk.tokenize import MWETokenizer
from gensim.corpora import WikiCorpus, wikicorpus, TextCorpus, MmCorpus, Dictionary
from gensim.models import LogEntropyModel
from gensim.similarities import Similarity
from nltk.tag.stanford import StanfordNERTagger

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




# -- COLLOCATION DETECTOR ------------------------------------------------

'''
The API consists of:
words2collocated(words):
    gets a list of words in string format and classifies them where necessary into expressions
    input:   a list of words in unicode/string format
    returns: a list of tokens in unicode/string format, either single words or collocations
'''

class Collocator():
    def __init__(self):
        self.bigram_measures  = BigramAssocMeasures()
        self.trigram_measures = TrigramAssocMeasures()


    # Takes a list of words and returns a list of tuples as collocation
    def words2collocations(self, words):
        self.bigram_finder  = BigramCollocationFinder.from_words(words, window_size = 20)
        self.trigram_finder = TrigramCollocationFinder.from_words(words, window_size = 20)

        l = int(len(words)/50)

        return self.bigram_finder.nbest(self.bigram_measures.pmi, l) + self.trigram_finder.nbest(self.bigram_measures.pmi, l)


    # Takes a list of words and returns a list of tokens either words or collocations
    def words2collocated(self, words):
        start = 0
        toks = []
        for ngram in self.words2collocations(words):
             toks.append(words[start:i+1])
             start = i+1
        if start < len(words):
            toks.append(words[start:])
        return toks



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
    # Helper function to generate the n-grams using  from the Brown
    # ngram is a function: nltk.bigram or nltk.trigram
    # AssocMeasures is a class: BigramAssocMeasures or TrigramAssocMeasures
    # CollocationFinder is a class: BigramCollocationFinder or TrigramCollocationFinder
    def __get_ngrams(self, training_sents, ngram, AssocMeasures, CollocationFinder):
        # to create sets of examples we use the bigrams we find in the training sentences
        ngrams = list(map(ngram, training_sents))
        ngrams = list(map(list, ngrams)) # unwrap ngrams generator objects
        ngrams = list(filter(lambda x: len(x) != 0, ngrams))

        # filter out punctuation from Brown
        corpus = list(filter(lambda x: x not in string.punctuation, nltk.corpus.brown.words()))
        finder = CollocationFinder.from_words(corpus, window_size = 20)
        # filter out stop words and infrequent collocations
        finder.apply_freq_filter(2)
        ignored_words = nltk.corpus.stopwords.words('english')
        finder.apply_word_filter(lambda w: w.lower() in ignored_words)

        found = finder.nbest(nltk.collocations.AssocMeasures.pmi, int(len(corpus)/50))
        return (ngrams, finder)


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


    # To train the classifier we use the Sharoff features on bigrams found in Brown
    # and also in the Sharoff Dictionary.
    # To create training examples, we use the features of bigrams above and for the
    # target whether they are in the set of bigram collocations in the Brown corpus
    # generated with the PMI measures provided by nltk's AssocMeasures()
    def __init__(self):
        print(str(datetime.now()) + ": Training decision tree classifier for the multi word expression tokenizer based on Sharoff's dictionary on the Brown corpus...")

        # get the Sharoff dictionary
        f = open("../corpora/sharoff.json", 'r+')
        self.SharoffDict = json.load(f)
        f.close()

        # get the corpus
        training_sents = [list(filter(lambda w: w not in string.punctuation, sent))
                          for sent in nltk.corpus.brown.sents()]
        # filter out empty or 1-word sentences
        training_sents = list(filter(lambda s: len(s) > 1, training_sents))
        toks = reduce(add, training_sents) # merge all sentences into one text of tokenized words

        # get all bigrams and trigrams in training_sents and all statistical bigram and trigram collocations found in treebank
        (bigrams,  finderbigrams)  = self.__get_ngrams(training_sents, nltk.bigrams,  BigramAssocMeasures,  BigramCollocationFinder)
        (trigrams, findertrigrams) = self.__get_ngrams(training_sents, nltk.trigrams, TrigramAssocMeasures, TrigramCollocationFinder)

        print(str(datetime.now()) + ": Constructing examples set with the Brown corpus...")
        # Create training examples with sharoff features, for each ngram in the dictionary
        examples = [(self.__Sharoff_features(expr), expr in finderbigrams or expr in findertrigrams)
                       for expr in bigrams or expr in trigrams
                       if ' '.join(map(str,expr)) in self.SharoffDict.keys()]

        print(str(datetime.now()) + ": Finished construction of " + len(examples) + " examples set with the Brown corpus.")
        #  classifier for training with the Brown corpus
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
                toks.append(words[start:i+1])
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
        f = open("../corpora/mccarthy.json", 'r+')
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
        self.chunker = nltk.data.load("chunkers/treebank_chunk_NaiveBayes.pickle")
        # self.tagger = nltk.data.load("taggers/brown_aubt.pickle")

    def sent2chunks(self, sentence):
        # Chose nltk.pos_tag for simplicty. For more complex answers, try brown
        # train chunker on brown if that's the case
        # tagged_sentences = list(map(self.tagger.tag, tokenized_sentences))
        tagged_sentence = nltk.pos_tag(tokenized_sentence)
        chunked_sentence = self.chunker.parse(tagged_sentence)
        return chunked_sentence

    def sents2chunks(self, tokenized_sentences):
        # Chose nltk.pos_tag for simplicty. For more complex answers, try brown
        # train chunker on brown if that's the case
        # tagged_sentences = list(map(self.tagger.tag, tokenized_sentences))
        tagged_sentences = list(map(nltk.pos_tag, tokenized_sentences))
        chunked_sentences = list(map(self.chunker.parse, tagged_sentences))
        return chunked_sentences




# -- TOPIC MODELLING ----------------------------------------------------------
'''
The TopicModelling class uses the latest Wikipedia dump as the main corpus and
dictionary, and for the intiial distribution of topics.
Uncomment the 'Wikipedia training' section whenever the corpus updates, then
re-run init.py to refresh the pickle.
'''
class TopicModelling():
    def __init__(self):
        wiki_src = '/disk100/wiki/xml/enwiki-articles.xml.bz2'
        dict_src = '/disk100/wiki/wiki_dictionary.dict'
        corp_src = '/disk100/wiki/wiki_corpus.mm'

        # -------------------------------------- Wikipedia training ------------------------------ #
        #                                                                                          #
        # print(str(datetime.now()) + ": Training gensim dictionary for the Wikipedia corpus...")  #
        #                                                                                          #
        # # load the corpus of documents in the wikipedia archive and save parsed files to disk    #
        # self.wiki_corpus = WikiCorpus(wiki_src)                                                  #
        # self.wiki_dictionary = self.wiki_corpus.dictionary                                       #
        # self.wiki_dictionary.save(dict_src)                                                      #
        # MmCorpus.serialize(corp_src, self.wiki_corpus)                                           #
        #                                                                                          #
        # print(str(datetime.now()) + ": Trained gensim dictionary for the Wikipedia corpus.")     #
        # ---------------------------------------------------------------------------------------- #

        # Working with persisted corpus and dictionary
#       print(str(datetime.now()) + ": Training gensim dictionary for the Wikipedia corpus...")

        # load the corpus of documents in the wikipedia archive and save parsed files to disk
#        self.wiki_corpus = WikiCorpus(wiki_src)
#        self.wiki_dictionary = self.wiki_corpus.dictionary
#        self.wiki_dictionary.save(dict_src)
#        MmCorpus.serialize(corp_src, self.wiki_corpus)

        # Working with persisted corpus and dictionary
        self.wiki_corpus = MmCorpus(corp_src)  # Revive a corpus using the Lazarus pit
        self.wiki_dictionary = Dictionary.load(dict_src)  # Load a dictionary

#       print(str(datetime.now()) + ": Trained gensim dictionary for the Wikipedia corpus.")


        # self.logent_transformation = LogEntropyModel(self.wiki_corpus, self.wiki_dictionary)
        # self.logent_corpus = MmCorpus(self.logent_transformation[self.wiki_corpus])
        # self.logent_transformation.save("/disk100/wiki/logent.model")
        # MmCorpus.serialize('/disk100/wiki/logent_corpus.mm', self.logent_corpus)
        # print(str(datetime.now()) + ": Trained logent corpus and transformation")

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

    def getsimilardocs(self, title, lsi_text, docs, index_documents, query, num):
      bow_doc = self.wiki_dictionary.doc2bow(wikicorpus.tokenize(lsi_text))
      logent_doc = self.logent_transformation[[bow_doc]]
      training_bow_docs = (self.wiki_dictionary.doc2bow(wikicorpus.tokenize(document))
                            for document in docs)
      logent_documents = self.logent_transformation[training_bow_docs]

      lsi_transformation = gensim.models.lsimodel.LsiModel(corpus=logent_corpus, id2word=self.wiki_dictionary, num_features=num)
      lsi_transformation.save("/disk100/wiki/lsi.model")

      corpus = (self.wiki_dictionary.doc2bow(wikicorpus.tokenize(doc))
                    for doc in index_documents)

      index = Similarity(corpus=lsi_transformation[logent_transformation[corpus]],
                        num_features=num, output_prefix="shard")

      sims_to_query = index[lsi_transformation[logent_transformation[self.wiki_dictionary.doc2bow(wikicorpus.tokenize(query))]]]
      best_score = max(sims_to_query)
      index = sims_to_query.tolist().index(best_score)
      most_similar_doc = docs[index]
      return most_similar_doc


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


class NERComboTagger(StanfordNERTagger):

  def __init__(self, *args, **kwargs):
    self.stanford_ner_models = kwargs['stanford_ner_models']
    kwargs.pop("stanford_ner_models")
    super(NERComboTagger,self).__init__(*args, **kwargs)

  @property
  def _cmd(self):
    return ['-mx7g',
            'edu.stanford.nlp.ie.NERClassifierCombiner',
            '-loadClassifier',
            self.stanford_ner_models,
            '-textFile',
            self._input_file_path,
            '-outputFormat',
            self._FORMAT,
            '-tokenizerFactory',
            'edu.stanford.nlp.process.WhitespaceTokenizer',
            '-tokenizerOptions',
            '\"tokenizeNLs=false\"']

class NameEntityDetector():
    def __init__(self):
        self.classifier_path1 = os.environ.get('STANFORD_MODELS') + '/seek5.ser.gz'
        self.name_tagger = NERComboTagger(self.classifier_path1,stanford_ner_models=self.classifier_path1)
        self.name_tagger._stanford_jar = os.environ.get("CLASSPATH")

    def text2ne(self, input_text):
        split_text = re.split("\,?\.?\s+", input_text)
        named_entities = dict(self.name_tagger.tag(split_text))
        return list(filter(lambda x: x[1] != 'O', named_entities.items()))

    def chunks2ne(self, input_text, chunked_sents):
        split_text = re.split("\,?\.?\s?\.?\,?", input_text)
        named_entities = dict(self.name_tagger.tag(split_text))
        named_entities = dict(filter(lambda x: x[1] != 'O', named_entities.items()))
        person_entities = dict(filter(lambda x: x[1] == 'PERSON', named_entities.items()))
        organization_entities = dict(filter(lambda x: x[1] == 'ORGANIZATION', named_entities.items()))
        location_entities = dict(filter(lambda x: x[1] == 'LOCATION', named_entities.items()))
        date_entities = dict(filter(lambda x: x[1] == 'DATE', named_entities.items()))
        # Create a list to store a more complete mapping of NEs
        answered = []
        multiple_entities = {   "PERSON" : [],
                                "DATE" : [],
                                "ORGANIZATION" : [],
                                "LOCATION" : []
                            }
        for chunked_sent in chunked_sents:
            filtered_chunked_subtrees = chunked_sent.subtrees(filter= lambda t: t.label() == 'NP')

            for subtree in filtered_chunked_subtrees:
                ent_key = []
                # Create Set like list to preserve order
                for t in subtree.leaves():
                    if (t[0] not in ent_key):
                        ent_key.append(t[0])
                if (all(word in ent_key for word in person_entities.keys())):
                    answered.append((ent_key, "PERSON"))
                if any(word in ent_key and word != 'the' for word in date_entities.keys()):
                    for word in ent_key:
                        if word in person_entities.keys():
                            multiple_entities["PERSON"].append(word)
                        if word in date_entities.keys():
                            multiple_entities["DATE"].append(word)
                        if word in organization_entities.keys():
                            multiple_entities["ORGANIZATION"].append(word)
                        if word in location_entities.keys():
                            multiple_entities["LOCATION"].append(word)
                    for key in multiple_entities.keys():
                        if len(multiple_entities[key]) > 0:
                            answered.append((multiple_entities[key], key))
                if any(word in ent_key for word in organization_entities.keys()):
                    answered.append((ent_key, "ORGANIZATION"))
                if any(word in ent_key for word in location_entities.keys()):
                    answered.append((ent_key, "LOCATION"))
        return list(filter(lambda x: x[1] != 'O' and x[1] != None, answered))

    def file2ne(self, src):
        p = subprocess.Popen(["java",
                                "-mx1g",
                                "edu.stanford.nlp.ie.NERClassifierCombiner",
                                "-loadClassifier",
                                self.classifier_path1,
                                "-textFile",
                                src,
                                "-outputFormat",
                                "tsv"], stdout=subprocess.PIPE)
        answer = []
        for l in p.stdout.read():
            answer.append(l + "\n")
        print(dict(filter(lambda x: x[1] != 'O', answer)))




# -- QUESTION CLASSIFIER --------------------------------------------------
'''
Using a modified version of the question classifier in nltk_data/corpora/qc

The classifier below tags a question with a semantic tag based on a set of
features: the presence of various words in the question's text, like any wh
word, the position of a question mark, the use of named entities and their
category etc.
It is trained with a modified version of the qc corpus, stored as dictionar
in ../corpora/, where each question is mapped to a tuple representing its
class and subclass

The API consists of:
classify(toks, nes):
    gets the tokenized text of the question and the named entities in it then classifies it accordingly
    toks:    a question as a list of tokens in unicode/string format
    nes:     the named entities that appear in the question
    returns: a tuple representing the class and subclass

'''

class QuestionClassifier():
    # extract question features from token list
    def __question_features(self, toks, nes):
        utoks = [w.lower() for w in toks]
        udict = dict(enumerate(utoks)).values()

        whlist = ['who', 'what', 'where', 'when', 'why', 'how']

        return {
            'is-who'  : 'who'   in udict or 'name'  in udict,
            'is-what' : 'what'  in udict or 'which' in udict,
            'is-where': 'where' in udict,
            'is-when' : 'when'  in udict or 'year'  in udict or 'time' in udict,
            'is-why'  : 'why'   in udict,
            'is-how'  : 'how'   in udict,
            'question-mark'      : '?' in udict,
            'question-mark-word' : '?' in udict and any(map(lambda x: x in udict and utoks.index(x) < utoks.index('?') or False, whlist)),
            'pers-ne' : any(map(lambda x: x[1] == 'PERSON',        nes)),
            'loc-ne'  : any(map(lambda x: x[1] == 'LOCATION' ,     nes)),
            'org-ne'  : any(map(lambda x: x[1] == 'ORGANIZATION' , nes)),
            'time-ne' : any(map(lambda x: x[1] == 'TIME',          nes)),
        }

    # Builds the classifier
    def __init__(self):
        print(str(datetime.now()) + ": Training question classifier with decision tree on modified QC corpus...")

        # get the question corpus into a set of tuples
        f = open("../corpora/qc.json", 'r+')
        training_qs = json.load(f)
        f.close()

        # Create training features by calling question_features on each tokenised question
        featuresets = [(self.__question_features(q[1][0], q[1][1]), ' '.join(q[1][2]))
                       for q in training_qs.items()]

        # Decision Tree classifier for training with the Treebank corpus
        size = int(len(featuresets)*0.2)
        train_set, test_set = featuresets[size:], featuresets[:size]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        print(str(datetime.now()) + ": Classifier trained with accuracy " + str(nltk.classify.accuracy(self.classifier, test_set)))


    # Classify questions
    def classify(self, toks, nes):
        return self.classifier.classify(self.__question_features(toks, nes))
