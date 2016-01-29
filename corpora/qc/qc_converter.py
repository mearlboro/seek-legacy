'''
quick script that generates the qc corpus with NE from the original nltk qc corpus

IMPORTANT:
to use move in /core
'''

import nltk
import json
from statistician import NameEntityDetector

f = open('../corpora/qc.txt')
lines = f.readlines()
toks = list(map(nltk.word_tokenize,lines))
tuples = [(line[3:],(line[0],line[3])) for line in toks]
triples = [(' '.join(pair[0]), (pair[0], ['NE'], pair[1])) for pair in tuples]
dictionary = dict(triples)

# TODO:
ner = NameEntityDetector()

with open('../corpora/qc.json', 'w') as out:
    json.dump(dictionary, out)


