import nltk
import string
import os
from nltk.corpus import wordnet
from nltk.collocations import *

# -- SENTENCE TOKENIZER ---------------------------------------------------
# "NLP with Python" book, chapter 6.2

# Tokenize text into words, punctuation, and spaces
class WordPunctSpaceTokenizer(nltk.tokenize.RegexpTokenizer):
    def __init__(self):
        nltk.tokenize.RegexpTokenizer.__init__(self, r'\w+|\s+|\.+|[\-\\=]+|[^\w\s]')

# Tokenize text into sentences
class SentenceTokenizer():
    # extract punctuation features from token list for punctuation sign (token of index i)
    def punctuation_features(self, toks, i):
        return {
            'punct': toks[i],
            'is-next-capitalized': (i < len(toks) - 1) and toks[i+1][0].isupper(),
            'lower-or-punct-prev': toks[i-1].lower() or toks[i-1] in string.punctuation,
            'is-prev-one-char': len(toks[i-1]) == 1
        }
 
    # Builds the classifier
    def __init__(self):
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
        featuresets = [(self.punctuation_features(toks,i), (i in bounds))
                       for i in range(1, len(toks)-1)
                       if toks[i] in '.?!']
 
        # Decision Tree classifier for training with the Treebank corpus
        size = int(len(featuresets)*0.2)
        train_set, test_set = featuresets[size:], featuresets[:size] 
        self.classifier = nltk.DecisionTreeClassifier.train(train_set)
        print(nltk.classify.accuracy(self.classifier, test_set))
 

    # Use the classifier to segment word toks into sentences
    def classify_sentences(self,words):
        start = 0
        sents = []
        for i in range(len(words)):
            if words[i] in '.?!' and self.classifier.classify(self.punctuation_features(words,i)) == True:
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            sents.append(words[start:])
        return sents
 

    # Segment text into sentences and words
    def segment_text(self, text):
        
        # turn whitespace characters into spaces: split() runs on whitespace then merge words back with spaces
        text = ' '.join(text.split())

        # tokenise with the Regexp tokenizer thus keeping the punctuation, words, and spaces
        toks = self.tokenizer.tokenize(text)
        # filter out irrelevant punctuation
        toks = list(filter(lambda tok: tok not in '"()[]{}', toks))

        # Create list of sentences using the classifier, then iterate through words in a sentence to collapse abbreviations into single words
        sentences = []
        for sent in self.classify_sentences(toks):
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




# -- MULTI WORD EXPRESSIONS CHUNKER ----------------------
# "NLP with Python" book, chapter 2.5
# using dictionaries found at mwe.stanford.edu/resources

class MultiWordExprChunker():

    # sharoff dictionary consists of a list of expressions, and their statistical collocation measures, the feature extractor below uses it to find out whether an expression is in the sharoff dictionary
    def Sharoff_features(self, expr):
        return {
            'expr': expr,
            'T-score':  self.SharoffDict[expr]['T']  or 0,
            'MI-score': self.SharoffDict[expr]['MI'] or 0,
            'LL-score': self.SharoffDict[expr]['LL'] or 0,
            'number-of-words': len(expr), 
        }
 
    # baldwin dictionary consists of a list of expressions, whether they are transitive or not, and their frequency 
    def Baldwin_features(self, sent, expr):
        return {
            'expr': expr,
            'transitive': self.BaldwinDict[expr][0] == is_transitive(sent, expr),
            'score':      self.BaldwinDict[expr][1] > 10,
            'number-of-words': len(expr),
        }
    # to find whether a verb is transitive, searches recursivly for 2 sibling nodes, one a VB and one an NP
    # def is_transitive(sent, verb):
    #     if  (len(tree) >= 2 and
    #          tree[0].node == 'VB' and
    #          tree[0,0] == verb and
    #      tree[1].node == 'NP'):      
    #             return 1 
    #     else:
    #         for child in tree:
    #              if isinstance(child, Tree):
    #                  if contains_trans(child,verb):
    #                      return 1
    #     return 0

    
    # Builds the classifier
    def __init__(self):
        # build the Sharoff and Baldwin dictionaries
        SharoffDict = {}
        f = open(r"../dictionaries/sharoff.txt")
        lines = f.readlines()
        f.close()
        i = 0
        currWordDict = {}
        for i in range(len(lines)):
            l = lines[i]
            items = l.split(': ')
            currWordDict[items[0]] = float(items[2][:-2])
            if i % 3 == 2:
                SharoffDict[items[1][:-1]] = currWordDict
                currWordDict = {}

        self.SharoffDict = SharoffDict

        BaldwinDict = []


        # join the sentence corpus into a text
        training_sents = nltk.corpus.treebank_raw.sents()
        toks = []
        bounds = set()
        offset = 0
        for sent in training_sents:
            sent = list(filter(lambda w:w not in ['START'] and w not in string.punctuation, sent))
            toks.extend(sent)  # union of toks in all sentences
            offset = offset + len(sent)  
            bounds.add(offset-1) # known boundaries of sentences 


        # to create sets of examples we use the collocations in treebank corpus
        bigrams = nltk.bigrams(toks)
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder2 = BigramCollocationFinder.from_words(
            nltk.corpus.treebank_raw.words(),
            window_size = 20)
        found2 = finder2.above_score(bigram_measures.raw_freq, 1.0 / len(tuple(nltk.bigrams(toks))))
        trigrams = nltk.trigrams(toks)
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder3 = TrigramCollocationFinder.from_words(
            nltk.corpus.treebank_raw.words(),
            window_size = 20)
        found3 = finder3.above_score(trigram_measures.raw_freq, 1.0 / len(tuple(nltk.trigrams(toks))))

        # Create training features with sharoff dictionary
        featuresets = [(self.Sharoff_features(expr), expr in found2 )
                       for expr in bigrams or expr in trigrams
                       if expr in self.SharoffDict.keys()]
 
        #  classifier for training with the Treebank corpus
        size = int(len(featuresets)*0.2)
        train_set, test_set = featuresets[size:], featuresets[:size] 
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(self.classifier, test_set))
 

    # Use the classifier to segment word toks into MWEs 
    def classify_mwe(self,words):
        start = 0
        toks = []
        for ngram in list(nltk.bigrams(words)) + list(nltk.trigrams(words)):
            if self.classifier.classify(self.Sharoff_features(ngram)) == True:
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            toks.append(words[start:])
        return toks
 




