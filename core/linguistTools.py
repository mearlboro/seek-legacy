import nltk
import string
 
# -- SENTENCE TOKENIZER ---------------------------------------------------
# "NLP with Python" book, chapter 6.2, pp234

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
 
        # Naive Bayes classifier for training with the Treebank corpus
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
